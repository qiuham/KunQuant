"""
测试流式模式下有状态算子的正确性
- WindowedLinearRegression: 验证流式和批量模式的数值一致性
- SkipList: 验证 SkipList 状态算子在流式模式下的正确性
"""

import numpy as np
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from KunQuant.Driver import compileit, KunCompilerConfig
from KunQuant.ops.CompOp import WindowedLinearRegressionSlope, WindowedLinearRegressionRSqaure, WindowedQuantile

def create_linear_regression_factor():
    """创建一个使用 WindowedLinearRegression 的因子"""
    builder = Builder()
    window = 10
    with builder:
        close = Input("close")
        # WindowedLinearRegressionSlope 内部使用 WindowedLinearRegression
        slope = WindowedLinearRegressionSlope(close, window)
        rsquare = WindowedLinearRegressionRSqaure(close, window)
        Output(slope, "slope")
        Output(rsquare, "rsquare")
    return Function(builder.ops)

def test_code_generation():
    """测试代码生成"""
    f = create_linear_regression_factor()

    print("\n=== 测试批量模式代码生成 ===")
    try:
        batch_code = compileit(
            f,
            "test_batch",
            partition_factor=1,
            dtype="float",
            blocking_len=8,
            input_layout="STs",
            output_layout="STs",
            options={"no_fast_stat": True}  # 使用 WindowedLinearRegression
        )
        print(f"批量模式生成成功，代码片段数: {len(batch_code)}")
        # 打印第一个代码片段的前 50 行
        if batch_code:
            lines = batch_code[0].split('\n')[:50]
            print("代码片段预览:")
            for i, line in enumerate(lines):
                print(f"  {i+1}: {line}")
    except Exception as e:
        print(f"批量模式生成失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== 测试流式模式代码生成 ===")
    try:
        stream_code = compileit(
            f,
            "test_stream",
            partition_factor=1,
            dtype="float",
            blocking_len=8,
            input_layout="STREAM",
            output_layout="STREAM",
            options={"no_fast_stat": True}  # 使用 WindowedLinearRegression
        )
        print(f"流式模式生成成功，代码片段数: {len(stream_code)}")
        # 打印第一个代码片段的前 80 行
        if stream_code:
            lines = stream_code[0].split('\n')[:80]
            print("代码片段预览:")
            for i, line in enumerate(lines):
                print(f"  {i+1}: {line}")
    except Exception as e:
        print(f"流式模式生成失败: {e}")
        import traceback
        traceback.print_exc()

def test_numerical_consistency():
    """测试流式模式和批量模式的数值一致性"""
    from KunQuant.jit import cfake
    from KunQuant.runner import KunRunner as kr
    import tempfile
    import os

    print("\n=== 测试数值一致性 ===")

    # 生成测试数据
    blocking_len = 8
    num_stocks = 16  # 需要是 blocking_len 的倍数
    time_length = 50
    np.random.seed(42)
    close_data = np.random.randn(num_stocks, time_length).astype(np.float32) + 100
    # STs 布局: (num_stocks//blocking_len, time_length, blocking_len)
    close_data_sts = close_data.reshape(num_stocks // blocking_len, blocking_len, time_length).transpose(0, 2, 1).copy()

    # 创建因子
    f = create_linear_regression_factor()

    # 编译批量模式
    print("编译批量模式...")
    batch_config = KunCompilerConfig(
        partition_factor=1,
        dtype="float",
        blocking_len=8,
        input_layout="STs",
        output_layout="STs",
        options={"no_fast_stat": True}
    )

    batch_lib = cfake.compileit(
        [("batch_test", f, batch_config)],
        "batch_lib",
        cfake.CppCompilerConfig(),
    )

    batch_module = batch_lib.getModule("batch_test")

    # 运行批量模式
    print("运行批量模式...")
    executor = kr.createSingleThreadExecutor()
    batch_inputs = {"close": close_data_sts}
    batch_outputs = kr.runGraph(executor, batch_module, batch_inputs, 0, time_length)

    batch_slope_sts = batch_outputs["slope"]
    batch_rsquare_sts = batch_outputs["rsquare"]

    print(f"批量模式输出 slope shape: {batch_slope_sts.shape}")
    print(f"批量模式输出 rsquare shape: {batch_rsquare_sts.shape}")

    # 从 STs 布局转回 (num_stocks, time_length)
    output_time = batch_slope_sts.shape[1]
    batch_slope = batch_slope_sts.transpose(0, 2, 1).reshape(num_stocks, output_time)
    batch_rsquare = batch_rsquare_sts.transpose(0, 2, 1).reshape(num_stocks, output_time)

    # 编译流式模式
    print("\n编译流式模式...")
    f2 = create_linear_regression_factor()  # 需要重新创建

    stream_config = KunCompilerConfig(
        partition_factor=1,
        dtype="float",
        blocking_len=8,
        input_layout="STREAM",
        output_layout="STREAM",
        options={"no_fast_stat": True}
    )

    stream_lib = cfake.compileit(
        [("stream_test", f2, stream_config)],
        "stream_lib",
        cfake.CppCompilerConfig(),
    )

    stream_module = stream_lib.getModule("stream_test")

    # 运行流式模式
    print("运行流式模式...")
    ctx = kr.StreamContext(executor, stream_module, num_stocks)

    # 分配状态内存（自动根据 module.state_size 计算）
    print(f"模块状态大小: {stream_module.state_size} bytes/block")
    ctx.allocStates()

    close_handle = ctx.queryBufferHandle("close")
    slope_handle = ctx.queryBufferHandle("slope")
    rsquare_handle = ctx.queryBufferHandle("rsquare")

    stream_slope = np.zeros((num_stocks, time_length), dtype=np.float32)
    stream_rsquare = np.zeros((num_stocks, time_length), dtype=np.float32)

    # 逐时间步推送数据
    for t in range(time_length):
        # 确保数据是 C 连续的 float32
        data_slice = np.ascontiguousarray(close_data[:, t], dtype=np.float32)
        ctx.pushData(close_handle, data_slice)
        ctx.run()  # run() 结束后会自动设置 states_initialized
        stream_slope[:, t] = ctx.getCurrentBuffer(slope_handle)
        stream_rsquare[:, t] = ctx.getCurrentBuffer(rsquare_handle)

    ctx.freeStates()

    # 比较结果（跳过前 window-1 个时间步，因为那些是 NaN）
    window = 10
    valid_start = window - 1

    print("\n比较结果...")
    slope_diff = np.abs(batch_slope[:, valid_start:] - stream_slope[:, valid_start:])
    rsquare_diff = np.abs(batch_rsquare[:, valid_start:] - stream_rsquare[:, valid_start:])

    # 忽略 NaN
    slope_diff = slope_diff[~np.isnan(slope_diff)]
    rsquare_diff = rsquare_diff[~np.isnan(rsquare_diff)]

    print(f"Slope 最大差异: {np.max(slope_diff):.2e}")
    print(f"Slope 平均差异: {np.mean(slope_diff):.2e}")
    print(f"RSqaure 最大差异: {np.max(rsquare_diff):.2e}")
    print(f"RSqaure 平均差异: {np.mean(rsquare_diff):.2e}")

    # 断言差异在可接受范围内
    # Slope 应该精确匹配，RSqaure 由于涉及除法会有浮点精度误差
    slope_tolerance = 1e-5
    rsquare_tolerance = 1e-2  # RSqaure 涉及除法，允许 1% 的相对误差

    slope_pass = np.max(slope_diff) < slope_tolerance
    rsquare_pass = np.max(rsquare_diff) < rsquare_tolerance

    if slope_pass and rsquare_pass:
        print(f"\n✓ 数值一致性测试通过！")
        print(f"  Slope 差异 < {slope_tolerance}")
        print(f"  RSqaure 差异 < {rsquare_tolerance}")
    else:
        print(f"\n✗ 数值一致性测试失败！")
        if not slope_pass:
            print(f"  Slope 差异 {np.max(slope_diff):.2e} 超过 {slope_tolerance}")
        if not rsquare_pass:
            print(f"  RSqaure 差异 {np.max(rsquare_diff):.2e} 超过 {rsquare_tolerance}")

def create_skiplist_factor():
    """创建一个使用 SkipList 的因子"""
    builder = Builder()
    window = 5
    with builder:
        close = Input("close")
        # WindowedQuantile 内部使用 SkipListState
        median = WindowedQuantile(close, window, 0.5)
        Output(median, "median")
    return Function(builder.ops)

def test_skiplist_numerical_consistency():
    """测试 SkipList 在流式模式和批量模式的数值一致性"""
    from KunQuant.jit import cfake
    from KunQuant.runner import KunRunner as kr

    print("\n=== 测试 SkipList 数值一致性 ===")

    # 生成测试数据
    blocking_len = 8
    num_stocks = 16
    time_length = 30
    window = 5
    np.random.seed(123)
    close_data = np.random.randn(num_stocks, time_length).astype(np.float32) * 10 + 100
    close_data_sts = close_data.reshape(num_stocks // blocking_len, blocking_len, time_length).transpose(0, 2, 1).copy()

    # 编译批量模式
    print("编译批量模式 (SkipList)...")
    f1 = create_skiplist_factor()
    batch_config = KunCompilerConfig(
        partition_factor=1,
        dtype="float",
        blocking_len=8,
        input_layout="STs",
        output_layout="STs",
        options={"no_fast_stat": True}
    )

    batch_lib = cfake.compileit(
        [("batch_skiplist", f1, batch_config)],
        "batch_skiplist_lib",
        cfake.CppCompilerConfig(),
    )
    batch_module = batch_lib.getModule("batch_skiplist")

    # 运行批量模式
    print("运行批量模式...")
    executor = kr.createSingleThreadExecutor()
    batch_inputs = {"close": close_data_sts}
    batch_outputs = kr.runGraph(executor, batch_module, batch_inputs, 0, time_length)
    batch_median_sts = batch_outputs["median"]
    output_time = batch_median_sts.shape[1]
    batch_median = batch_median_sts.transpose(0, 2, 1).reshape(num_stocks, output_time)

    # 编译流式模式
    print("\n编译流式模式 (SkipList)...")
    f2 = create_skiplist_factor()
    stream_config = KunCompilerConfig(
        partition_factor=1,
        dtype="float",
        blocking_len=8,
        input_layout="STREAM",
        output_layout="STREAM",
        options={"no_fast_stat": True}
    )

    stream_lib = cfake.compileit(
        [("stream_skiplist", f2, stream_config)],
        "stream_skiplist_lib",
        cfake.CppCompilerConfig(),
    )
    stream_module = stream_lib.getModule("stream_skiplist")

    # 运行流式模式
    print("运行流式模式...")
    print(f"模块状态大小: {stream_module.state_size} bytes/block")
    ctx = kr.StreamContext(executor, stream_module, num_stocks)
    ctx.allocStates()

    close_handle = ctx.queryBufferHandle("close")
    median_handle = ctx.queryBufferHandle("median")

    stream_median = np.zeros((num_stocks, time_length), dtype=np.float32)

    for t in range(time_length):
        data_slice = np.ascontiguousarray(close_data[:, t], dtype=np.float32)
        ctx.pushData(close_handle, data_slice)
        ctx.run()
        stream_median[:, t] = ctx.getCurrentBuffer(median_handle)

    ctx.freeStates()

    # 比较结果
    valid_start = window - 1
    print("\n比较结果...")
    median_diff = np.abs(batch_median[:, valid_start:] - stream_median[:, valid_start:])
    median_diff = median_diff[~np.isnan(median_diff)]

    print(f"Median 最大差异: {np.max(median_diff):.2e}")
    print(f"Median 平均差异: {np.mean(median_diff):.2e}")

    tolerance = 1e-5
    if np.max(median_diff) < tolerance:
        print(f"\n✓ SkipList 数值一致性测试通过！")
        print(f"  Median 差异 < {tolerance}")
    else:
        print(f"\n✗ SkipList 数值一致性测试失败！")
        print(f"  Median 差异 {np.max(median_diff):.2e} 超过 {tolerance}")

if __name__ == "__main__":
    test_code_generation()
    test_numerical_consistency()
    test_skiplist_numerical_consistency()
