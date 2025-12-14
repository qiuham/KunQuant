"""
测试 WindowedMax, WindowedMin, WindowedStddev, WindowedVar 在流式模式下的正确性
"""

import numpy as np
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from KunQuant.Driver import KunCompilerConfig
from KunQuant.ops.CompOp import WindowedMax, WindowedMin, WindowedVar, WindowedStddev, WindowedAvg


def create_windowed_ops_factor(window=5):
    """创建包含多种窗口操作的因子"""
    builder = Builder()
    with builder:
        close = Input("close")
        Output(WindowedMax(close, window), "max")
        Output(WindowedMin(close, window), "min")
        Output(WindowedVar(close, window), "var")
        Output(WindowedStddev(close, window), "std")
        Output(WindowedAvg(close, window), "avg")
    return Function(builder.ops)


def test_windowed_ops_stream():
    """测试流式模式下窗口操作的正确性"""
    from KunQuant.jit import cfake
    from KunQuant.runner import KunRunner as kr
    import pandas as pd

    print("\n=== 测试窗口操作流式模式 ===")

    # 参数
    blocking_len = 8
    num_stocks = 16
    time_length = 30
    window = 5

    np.random.seed(42)
    # 生成 (num_stocks, time_length) 的数据
    close_data = np.random.randn(num_stocks, time_length).astype(np.float32) + 100

    # 计算 pandas 参考值 (每个 stock 独立计算)
    expected = {}
    for i in range(num_stocks):
        df = pd.Series(close_data[i])
        expected.setdefault('max', []).append(df.rolling(window).max().values)
        expected.setdefault('min', []).append(df.rolling(window).min().values)
        expected.setdefault('var', []).append(df.rolling(window).var().values)
        expected.setdefault('std', []).append(df.rolling(window).std().values)
        expected.setdefault('avg', []).append(df.rolling(window).mean().values)

    for k in expected:
        expected[k] = np.array(expected[k])  # (num_stocks, time_length)

    # ========== 批量模式 ==========
    print("\n--- 批量模式 ---")
    f_batch = create_windowed_ops_factor(window)

    batch_config = KunCompilerConfig()
    batch_config.partition_factor = 1
    batch_config.dtype = "float"
    batch_config.blocking_len = blocking_len
    batch_config.input_layout = "STs"
    batch_config.output_layout = "STs"

    batch_lib = cfake.compileit(
        [("batch_windowed", f_batch, batch_config)],
        "batch_windowed_lib",
        cfake.CppCompilerConfig(),
    )
    batch_module = batch_lib.getModule("batch_windowed")

    # STs 布局转换
    close_sts = close_data.reshape(num_stocks // blocking_len, blocking_len, time_length).transpose(0, 2, 1).copy()

    executor = kr.createSingleThreadExecutor()
    batch_outputs = kr.runGraph(executor, batch_module, {"close": close_sts}, 0, time_length)

    # 转回 (num_stocks, time_length)
    batch_results = {}
    for name in ['max', 'min', 'var', 'std', 'avg']:
        out_sts = batch_outputs[name]
        batch_results[name] = out_sts.transpose(0, 2, 1).reshape(num_stocks, -1)

    print("批量模式输出 shape:", batch_results['max'].shape)

    # 验证批量模式
    for name in ['max', 'min', 'var', 'std', 'avg']:
        batch_out = batch_results[name]
        exp = expected[name]
        # 只比较非 NaN 部分
        mask = ~np.isnan(exp)
        if np.sum(mask) > 0:
            match = np.allclose(batch_out[mask], exp[mask], rtol=1e-4, atol=1e-6)
            print(f"  批量 {name}: {'PASS' if match else 'FAIL'}")
            if not match:
                diff = np.abs(batch_out - exp)
                diff[np.isnan(diff)] = 0
                max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"    最大差异位置: {max_diff_idx}, batch={batch_out[max_diff_idx]:.6f}, expected={exp[max_diff_idx]:.6f}")

    # ========== 流式模式 ==========
    print("\n--- 流式模式 ---")
    f_stream = create_windowed_ops_factor(window)

    stream_config = KunCompilerConfig()
    stream_config.partition_factor = 1
    stream_config.dtype = "float"
    stream_config.blocking_len = blocking_len
    stream_config.input_layout = "STREAM"
    stream_config.output_layout = "STREAM"

    stream_lib = cfake.compileit(
        [("stream_windowed", f_stream, stream_config)],
        "stream_windowed_lib",
        cfake.CppCompilerConfig(),
    )
    stream_module = stream_lib.getModule("stream_windowed")

    # 流式执行
    ctx = kr.StreamContext(executor, stream_module, num_stocks)
    ctx.allocStates()

    # 获取 buffer 句柄
    close_handle = ctx.queryBufferHandle("close")
    max_handle = ctx.queryBufferHandle("max")
    min_handle = ctx.queryBufferHandle("min")
    var_handle = ctx.queryBufferHandle("var")
    std_handle = ctx.queryBufferHandle("std")
    avg_handle = ctx.queryBufferHandle("avg")

    stream_results = {name: [] for name in ['max', 'min', 'var', 'std', 'avg']}

    for t in range(time_length):
        # 推入数据
        ctx.pushData(close_handle, close_data[:, t].copy())
        # 运行
        ctx.run()
        # 获取输出
        stream_results['max'].append(np.array(ctx.getCurrentBuffer(max_handle)).copy())
        stream_results['min'].append(np.array(ctx.getCurrentBuffer(min_handle)).copy())
        stream_results['var'].append(np.array(ctx.getCurrentBuffer(var_handle)).copy())
        stream_results['std'].append(np.array(ctx.getCurrentBuffer(std_handle)).copy())
        stream_results['avg'].append(np.array(ctx.getCurrentBuffer(avg_handle)).copy())

    # 转换为 (num_stocks, time_length)
    for name in stream_results:
        stream_results[name] = np.array(stream_results[name]).T

    print("流式模式输出 shape:", stream_results['max'].shape)

    # 验证流式模式
    all_pass = True
    for name in ['max', 'min', 'var', 'std', 'avg']:
        stream_out = stream_results[name]
        exp = expected[name]
        mask = ~np.isnan(exp)
        if np.sum(mask) > 0:
            match = np.allclose(stream_out[mask], exp[mask], rtol=1e-3, atol=1e-4)
            print(f"  流式 {name}: {'PASS' if match else 'FAIL'}")
            if not match:
                all_pass = False
                diff = np.abs(stream_out - exp)
                diff[np.isnan(diff)] = 0
                max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"    最大差异位置: {max_diff_idx}, stream={stream_out[max_diff_idx]:.6f}, expected={exp[max_diff_idx]:.6f}")
                # 打印前几个值
                print(f"    流式输出前10个 (stock 0): {stream_out[0, :10]}")
                print(f"    期望值前10个 (stock 0):   {exp[0, :10]}")

    return all_pass


if __name__ == "__main__":
    success = test_windowed_ops_stream()
    print("\n" + "="*50)
    print(f"测试结果: {'全部通过' if success else '存在失败'}")
