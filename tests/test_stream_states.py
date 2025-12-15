"""
Test stateful operators in stream mode
- WindowedLinearRegression: verify numerical consistency between stream and batch modes
- SkipList: verify SkipList stateful operator correctness in stream mode
"""

import numpy as np
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from KunQuant.Driver import compileit, KunCompilerConfig
from KunQuant.ops.CompOp import WindowedLinearRegressionSlope, WindowedLinearRegressionRSqaure, WindowedQuantile
from KunQuant.ops.MiscOp import ExpMovingAvg

def create_linear_regression_factor():
    """Create a factor using WindowedLinearRegression"""
    builder = Builder()
    window = 10
    with builder:
        close = Input("close")
        # WindowedLinearRegressionSlope internally uses WindowedLinearRegression
        slope = WindowedLinearRegressionSlope(close, window)
        rsquare = WindowedLinearRegressionRSqaure(close, window)
        Output(slope, "slope")
        Output(rsquare, "rsquare")
    return Function(builder.ops)

def test_code_generation():
    """Test code generation"""
    f = create_linear_regression_factor()

    print("\n=== Test batch mode code generation ===")
    try:
        batch_code = compileit(
            f,
            "test_batch",
            partition_factor=1,
            dtype="float",
            blocking_len=8,
            input_layout="STs",
            output_layout="STs",
            options={"no_fast_stat": True}  # Use WindowedLinearRegression
        )
        print(f"Batch mode generation succeeded, code snippets: {len(batch_code)}")
        # Print first 50 lines of the first code snippet
        if batch_code:
            lines = batch_code[0].split('\n')[:50]
            print("Code snippet preview:")
            for i, line in enumerate(lines):
                print(f"  {i+1}: {line}")
    except Exception as e:
        print(f"Batch mode generation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Test stream mode code generation ===")
    try:
        stream_code = compileit(
            f,
            "test_stream",
            partition_factor=1,
            dtype="float",
            blocking_len=8,
            input_layout="STREAM",
            output_layout="STREAM",
            options={"no_fast_stat": True}  # Use WindowedLinearRegression
        )
        print(f"Stream mode generation succeeded, code snippets: {len(stream_code)}")
        # Print first 80 lines of the first code snippet
        if stream_code:
            lines = stream_code[0].split('\n')[:80]
            print("Code snippet preview:")
            for i, line in enumerate(lines):
                print(f"  {i+1}: {line}")
    except Exception as e:
        print(f"Stream mode generation failed: {e}")
        import traceback
        traceback.print_exc()

def test_numerical_consistency():
    """Test numerical consistency between stream and batch modes"""
    from KunQuant.jit import cfake
    from KunQuant.runner import KunRunner as kr
    import tempfile
    import os

    print("\n=== Test numerical consistency ===")

    # Generate test data
    blocking_len = 8
    num_stocks = 16  # Must be multiple of blocking_len
    time_length = 50
    np.random.seed(42)
    close_data = np.random.randn(num_stocks, time_length).astype(np.float32) + 100
    # STs layout: (num_stocks//blocking_len, time_length, blocking_len)
    close_data_sts = close_data.reshape(num_stocks // blocking_len, blocking_len, time_length).transpose(0, 2, 1).copy()

    # Create factor
    f = create_linear_regression_factor()

    # Compile batch mode
    print("Compiling batch mode...")
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

    # Run batch mode
    print("Running batch mode...")
    executor = kr.createSingleThreadExecutor()
    batch_inputs = {"close": close_data_sts}
    batch_outputs = kr.runGraph(executor, batch_module, batch_inputs, 0, time_length)

    batch_slope_sts = batch_outputs["slope"]
    batch_rsquare_sts = batch_outputs["rsquare"]

    print(f"Batch mode output slope shape: {batch_slope_sts.shape}")
    print(f"Batch mode output rsquare shape: {batch_rsquare_sts.shape}")

    # Convert from STs layout back to (num_stocks, time_length)
    output_time = batch_slope_sts.shape[1]
    batch_slope = batch_slope_sts.transpose(0, 2, 1).reshape(num_stocks, output_time)
    batch_rsquare = batch_rsquare_sts.transpose(0, 2, 1).reshape(num_stocks, output_time)

    # Compile stream mode
    print("\nCompiling stream mode...")
    f2 = create_linear_regression_factor()  # Need to recreate

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

    # Run stream mode
    print("Running stream mode...")
    ctx = kr.StreamContext(executor, stream_module, num_stocks)

    # Allocate state memory (auto-calculated from module.state_size)
    print(f"Module state size: {stream_module.state_size} bytes/block")
    ctx.allocStates()

    close_handle = ctx.queryBufferHandle("close")
    slope_handle = ctx.queryBufferHandle("slope")
    rsquare_handle = ctx.queryBufferHandle("rsquare")

    stream_slope = np.zeros((num_stocks, time_length), dtype=np.float32)
    stream_rsquare = np.zeros((num_stocks, time_length), dtype=np.float32)

    # Push data step by step
    for t in range(time_length):
        # Ensure data is C-contiguous float32
        data_slice = np.ascontiguousarray(close_data[:, t], dtype=np.float32)
        ctx.pushData(close_handle, data_slice)
        ctx.run()  # run() auto-sets states_initialized after completion
        stream_slope[:, t] = ctx.getCurrentBuffer(slope_handle)
        stream_rsquare[:, t] = ctx.getCurrentBuffer(rsquare_handle)

    ctx.freeStates()

    # Compare results (skip first window-1 timesteps as those are NaN)
    window = 10
    valid_start = window - 1

    print("\nComparing results...")
    slope_diff = np.abs(batch_slope[:, valid_start:] - stream_slope[:, valid_start:])
    rsquare_diff = np.abs(batch_rsquare[:, valid_start:] - stream_rsquare[:, valid_start:])

    # Ignore NaN
    slope_diff = slope_diff[~np.isnan(slope_diff)]
    rsquare_diff = rsquare_diff[~np.isnan(rsquare_diff)]

    print(f"Slope max diff: {np.max(slope_diff):.2e}")
    print(f"Slope avg diff: {np.mean(slope_diff):.2e}")
    print(f"RSqaure max diff: {np.max(rsquare_diff):.2e}")
    print(f"RSqaure avg diff: {np.mean(rsquare_diff):.2e}")

    # Assert difference is within acceptable range
    # Slope should match exactly, RSqaure has floating point error due to division
    slope_tolerance = 1e-5
    rsquare_tolerance = 1e-2  # RSqaure involves division, allow 1% relative error

    slope_pass = np.max(slope_diff) < slope_tolerance
    rsquare_pass = np.max(rsquare_diff) < rsquare_tolerance

    if slope_pass and rsquare_pass:
        print(f"\n✓ Numerical consistency test passed!")
        print(f"  Slope diff < {slope_tolerance}")
        print(f"  RSqaure diff < {rsquare_tolerance}")
        return True
    else:
        print(f"\n✗ Numerical consistency test failed!")
        if not slope_pass:
            print(f"  Slope diff {np.max(slope_diff):.2e} exceeds {slope_tolerance}")
        if not rsquare_pass:
            print(f"  RSqaure diff {np.max(rsquare_diff):.2e} exceeds {rsquare_tolerance}")
        return False

def create_skiplist_factor():
    """Create a factor using SkipList"""
    builder = Builder()
    window = 5
    with builder:
        close = Input("close")
        # WindowedQuantile internally uses SkipListState
        median = WindowedQuantile(close, window, 0.5)
        Output(median, "median")
    return Function(builder.ops)

def test_skiplist_numerical_consistency():
    """Test SkipList numerical consistency between stream and batch modes"""
    from KunQuant.jit import cfake
    from KunQuant.runner import KunRunner as kr

    print("\n=== Test SkipList numerical consistency ===")

    # Generate test data
    blocking_len = 8
    num_stocks = 16
    time_length = 30
    window = 5
    np.random.seed(123)
    close_data = np.random.randn(num_stocks, time_length).astype(np.float32) * 10 + 100
    close_data_sts = close_data.reshape(num_stocks // blocking_len, blocking_len, time_length).transpose(0, 2, 1).copy()

    # Compile batch mode
    print("Compiling batch mode (SkipList)...")
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

    # Run batch mode
    print("Running batch mode...")
    executor = kr.createSingleThreadExecutor()
    batch_inputs = {"close": close_data_sts}
    batch_outputs = kr.runGraph(executor, batch_module, batch_inputs, 0, time_length)
    batch_median_sts = batch_outputs["median"]
    output_time = batch_median_sts.shape[1]
    batch_median = batch_median_sts.transpose(0, 2, 1).reshape(num_stocks, output_time)

    # Compile stream mode
    print("\nCompiling stream mode (SkipList)...")
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

    # Run stream mode
    print("Running stream mode...")
    print(f"Module state size: {stream_module.state_size} bytes/block")
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

    # Compare results
    valid_start = window - 1
    print("\nComparing results...")
    median_diff = np.abs(batch_median[:, valid_start:] - stream_median[:, valid_start:])
    median_diff = median_diff[~np.isnan(median_diff)]

    print(f"Median max diff: {np.max(median_diff):.2e}")
    print(f"Median avg diff: {np.mean(median_diff):.2e}")

    tolerance = 1e-5
    if np.max(median_diff) < tolerance:
        print(f"\n✓ SkipList numerical consistency test passed!")
        print(f"  Median diff < {tolerance}")
        return True
    else:
        print(f"\n✗ SkipList numerical consistency test failed!")
        print(f"  Median diff {np.max(median_diff):.2e} exceeds {tolerance}")
        return False

def test_reset_states():
    """Test resetStates: should get same results after reset"""
    from KunQuant.jit import cfake
    from KunQuant.runner import KunRunner as kr

    print("\n=== Test resetStates functionality ===")

    # Generate test data
    blocking_len = 8
    num_stocks = 16
    time_length = 20
    window = 5
    np.random.seed(456)
    close_data = np.random.randn(num_stocks, time_length).astype(np.float32) * 10 + 100

    # Compile stream mode
    f = create_skiplist_factor()
    stream_config = KunCompilerConfig(
        partition_factor=1,
        dtype="float",
        blocking_len=8,
        input_layout="STREAM",
        output_layout="STREAM",
        options={"no_fast_stat": True}
    )

    stream_lib = cfake.compileit(
        [("reset_test", f, stream_config)],
        "reset_test_lib",
        cfake.CppCompilerConfig(),
    )
    stream_module = stream_lib.getModule("reset_test")

    # Create StreamContext
    executor = kr.createSingleThreadExecutor()
    ctx = kr.StreamContext(executor, stream_module, num_stocks)
    ctx.allocStates()

    close_handle = ctx.queryBufferHandle("close")
    median_handle = ctx.queryBufferHandle("median")

    # First run
    print("First run...")
    result1 = np.zeros((num_stocks, time_length), dtype=np.float32)
    for t in range(time_length):
        data_slice = np.ascontiguousarray(close_data[:, t], dtype=np.float32)
        ctx.pushData(close_handle, data_slice)
        ctx.run()
        result1[:, t] = ctx.getCurrentBuffer(median_handle)

    # Reset states
    print("Resetting states...")
    ctx.resetStates()

    # Second run (same data)
    print("Second run (same data)...")
    result2 = np.zeros((num_stocks, time_length), dtype=np.float32)
    for t in range(time_length):
        data_slice = np.ascontiguousarray(close_data[:, t], dtype=np.float32)
        ctx.pushData(close_handle, data_slice)
        ctx.run()
        result2[:, t] = ctx.getCurrentBuffer(median_handle)

    ctx.freeStates()

    # Compare two runs
    valid_start = window - 1
    diff = np.abs(result1[:, valid_start:] - result2[:, valid_start:])
    diff = diff[~np.isnan(diff)]

    print(f"Max diff between two runs: {np.max(diff):.2e}")

    if np.max(diff) < 1e-6:
        print(f"\n✓ resetStates test passed! Results consistent after reset")
        return True
    else:
        print(f"\n✗ resetStates test failed! Results inconsistent")
        return False

def create_ema_factor():
    """Create a factor using ExpMovingAvg"""
    builder = Builder()
    window = 10
    with builder:
        close = Input("close")
        ema = ExpMovingAvg(close, window)
        Output(ema, "ema")
    return Function(builder.ops)

def test_ema_numerical_consistency():
    """Test ExpMovingAvg numerical consistency between stream and batch modes"""
    from KunQuant.jit import cfake
    from KunQuant.runner import KunRunner as kr

    print("\n=== Test ExpMovingAvg numerical consistency ===")

    # Generate test data
    blocking_len = 8
    num_stocks = 16
    time_length = 30
    np.random.seed(789)
    close_data = np.random.randn(num_stocks, time_length).astype(np.float32) * 10 + 100
    close_data_sts = close_data.reshape(num_stocks // blocking_len, blocking_len, time_length).transpose(0, 2, 1).copy()

    # Compile batch mode
    print("Compiling batch mode (EMA)...")
    f1 = create_ema_factor()
    batch_config = KunCompilerConfig(
        partition_factor=1,
        dtype="float",
        blocking_len=8,
        input_layout="STs",
        output_layout="STs",
        options={}
    )

    batch_lib = cfake.compileit(
        [("batch_ema", f1, batch_config)],
        "batch_ema_lib",
        cfake.CppCompilerConfig(),
    )
    batch_module = batch_lib.getModule("batch_ema")

    # Run batch mode
    print("Running batch mode...")
    executor = kr.createSingleThreadExecutor()
    batch_inputs = {"close": close_data_sts}
    batch_outputs = kr.runGraph(executor, batch_module, batch_inputs, 0, time_length)
    batch_ema_sts = batch_outputs["ema"]
    output_time = batch_ema_sts.shape[1]
    batch_ema = batch_ema_sts.transpose(0, 2, 1).reshape(num_stocks, output_time)

    # Compile stream mode
    print("\nCompiling stream mode (EMA)...")
    f2 = create_ema_factor()
    stream_config = KunCompilerConfig(
        partition_factor=1,
        dtype="float",
        blocking_len=8,
        input_layout="STREAM",
        output_layout="STREAM",
        options={}
    )

    stream_lib = cfake.compileit(
        [("stream_ema", f2, stream_config)],
        "stream_ema_lib",
        cfake.CppCompilerConfig(),
    )
    stream_module = stream_lib.getModule("stream_ema")

    # Run stream mode
    print("Running stream mode...")
    print(f"Module state size: {stream_module.state_size} bytes/block")
    ctx = kr.StreamContext(executor, stream_module, num_stocks)
    ctx.allocStates()

    close_handle = ctx.queryBufferHandle("close")
    ema_handle = ctx.queryBufferHandle("ema")

    stream_ema = np.zeros((num_stocks, time_length), dtype=np.float32)

    for t in range(time_length):
        data_slice = np.ascontiguousarray(close_data[:, t], dtype=np.float32)
        ctx.pushData(close_handle, data_slice)
        ctx.run()
        stream_ema[:, t] = ctx.getCurrentBuffer(ema_handle)

    ctx.freeStates()

    # Compare results (EMA has output from first timestep)
    print("\nComparing results...")
    ema_diff = np.abs(batch_ema - stream_ema)
    ema_diff = ema_diff[~np.isnan(ema_diff)]

    print(f"EMA max diff: {np.max(ema_diff):.2e}")
    print(f"EMA avg diff: {np.mean(ema_diff):.2e}")

    tolerance = 1e-5
    if np.max(ema_diff) < tolerance:
        print(f"\n✓ ExpMovingAvg numerical consistency test passed!")
        print(f"  EMA diff < {tolerance}")
        return True
    else:
        print(f"\n✗ ExpMovingAvg numerical consistency test failed!")
        print(f"  EMA diff {np.max(ema_diff):.2e} exceeds {tolerance}")
        return False

if __name__ == "__main__":
    import sys
    test_code_generation()
    results = [
        test_numerical_consistency(),
        test_skiplist_numerical_consistency(),
        test_reset_states(),
        test_ema_numerical_consistency()
    ]
    all_pass = all(results)
    print("\n" + "="*50)
    print(f"Test result: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
