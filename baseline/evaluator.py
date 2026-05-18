import subprocess
import os
import sys
from glob import glob
import json
import time
from datetime import datetime
from utils import Colors, log_subprocess_result, next_task_log_dir, relpath, terminal_log, to_text, write_run_summary, write_task_summary

# Default timeout settings (seconds) for each task, can be customized per task
DEFAULT_RUN_TIMEOUTS = {
    "sort": 600,
    "cabinet": 600,
    "rope": 600,
    "sweep": 600,
    "sandwich": 600,
    "pack": 600,
}

def test_run_dialog(task: str, num_runs: int, output_dir: str, seed: int = 0, run_timeout: float = None):
    """
    Test and run dialog tasks
    
    Args:
        task: Task name
        num_runs: Number of runs
        output_dir: Output directory
        seed: Random seed
        run_timeout: Timeout for single run (seconds). If None, use value from DEFAULT_RUN_TIMEOUTS, default 60s
    """
    # If timeout not specified, use task default or global default of 60s
    if run_timeout is None:
        run_timeout = DEFAULT_RUN_TIMEOUTS.get(task, 60)
    
    print("\n" + Colors.CYAN + Colors.BOLD + f"▶ Starting Task: {task.upper()}" + Colors.ENDC)
    print(Colors.CYAN + f"  Configuration: {num_runs} runs, {run_timeout}s timeout per run" + Colors.ENDC)
    
    task_log_dir = next_task_log_dir(task)
    if task_log_dir is not None:
        task_output_dir = task_log_dir
        run_name = "runs"
        artifacts_dir = os.path.join(task_output_dir, run_name)
        os.makedirs(artifacts_dir, exist_ok=True)
        print(Colors.CYAN + f"  Task artifacts: {relpath(artifacts_dir)}" + Colors.ENDC)
    else:
        task_output_dir = output_dir
        run_name = task
        artifacts_dir = os.path.join(task_output_dir, run_name)

    # Record number of runs before execution, to only count runs from this execution
    existing_runs = glob(os.path.join(task_output_dir, run_name, 'run_*'))
    start_run_count = len(existing_runs)
    if start_run_count > 0:
        existing_run_ids = [int(run.split("_")[-1]) for run in existing_runs]
        next_run_id = max(existing_run_ids) + 1
    else:
        next_run_id = 0
    
    # Record start time
    start_time = time.time()
    
    # Calculate total timeout: number of runs * timeout per run, with some buffer (20%)
    total_timeout = num_runs * run_timeout * 1.2
    
    command = [sys.executable, 'run_dialog.py', '--task', task, '--run_name', run_name,
               '--data_dir', task_output_dir,
               '--start_id', str(-1),
               '--num_runs', str(num_runs),
               '--skip_display',
               '--tsteps', str(10),
               '--seed', str(seed),
               '--run_timeout', str(run_timeout)]  # Pass run timeout parameter

    subprocess_started_at = datetime.now()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=total_timeout)
    except subprocess.TimeoutExpired as e:
        print(Colors.RED + f"\n✗ Subprocess timed out after {total_timeout:.0f}s" + Colors.ENDC)
        result = subprocess.CompletedProcess(
            args=e.args,
            returncode=-1,
            stdout=to_text(e.stdout),
            stderr=f"Subprocess timeout after {total_timeout:.0f}s\n" + to_text(e.stderr)
        )
    subprocess_finished_at = datetime.now()

    task_log_dir = log_subprocess_result(
        task,
        result,
        command,
        started_at=subprocess_started_at,
        finished_at=subprocess_finished_at,
        task_dir=task_log_dir,
    )
    
    total_time = time.time() - start_time
    
    success_cnt = 0
    total_cnt = 0
    total_steps = 0
    timeout_cnt = 0

    # Only count runs from this execution (num_runs starting from next_run_id)
    current_run_ids = list(range(next_run_id, next_run_id + num_runs))
    
    for run_id in current_run_ids:
        run_dir = os.path.join(task_output_dir, run_name, f'run_{run_id}')
        if not os.path.exists(run_dir):
            print(f"Warning: Run {run_id} directory not found")
            continue
            
        all_jsons = glob(os.path.join(run_dir, "*.json"))
        if len(all_jsons) == 0:
            print(f"Warning: Run {run_id} has no json files")
            continue
            
        json_dir = all_jsons[0]
        with open(json_dir, 'r', encoding='utf8') as fp:
            json_data = json.load(fp)
            success = json_data.get('success', False)
            steps = json_data.get('step', 0)
            timed_out = json_data.get('timed_out', False)
            
            if timed_out:
                timeout_cnt += 1
            
            if success:
                success_cnt += 1
                total_steps += steps
        
        total_cnt += 1

    if total_cnt < num_runs:
        print(Colors.YELLOW + f"Warning: Task {task} completed {total_cnt}/{num_runs} runs" + Colors.ENDC)

    print("\n" + Colors.BOLD + f"▶ RESULTS FOR TASK: {task.upper()}" + Colors.ENDC)
    
    # Success metrics
    success_pct = 100 * success_cnt / total_cnt if total_cnt > 0 else 0
    if success_pct >= 80:
        color = Colors.GREEN
    elif success_pct >= 50:
        color = Colors.YELLOW
    else:
        color = Colors.RED
    print("Success Rate:  " + color + f"{success_cnt}/{total_cnt} ({success_pct:.1f}%)" + Colors.ENDC)
    
    # Timeout metrics
    if timeout_cnt > 0:
        print("Timeout Count: " + Colors.YELLOW + f"{timeout_cnt}/{total_cnt} ({100*timeout_cnt/total_cnt:.1f}%)" + Colors.ENDC)
    else:
        print(f"Timeout Count: {timeout_cnt}/{total_cnt}")
    
    # Steps metrics
    if success_cnt > 0:
        avg_steps = total_steps / success_cnt
        print("Average Steps: " + Colors.CYAN + f"{avg_steps:.2f}" + Colors.ENDC + " (successful runs only)")
    else:
        print("Average Steps: N/A (no successful runs)")
    
    # Time metrics
    print("Total Time:    " + Colors.BLUE + f"{total_time:.2f}s ({total_time/60:.2f} minutes)" + Colors.ENDC)
    
    # Return code
    if result.returncode == 0:
        print("Return Code:   " + Colors.GREEN + f"{result.returncode}" + Colors.ENDC + " (Success)")
    else:
        print("Return Code:   " + Colors.RED + f"{result.returncode}" + Colors.ENDC + " (Error)")
    
    # Error output if any
    if result.stderr and result.stderr.strip():
        print(Colors.YELLOW + "\nStderr Output:" + Colors.ENDC)
        stderr_lines = result.stderr[:500].split('\n')
        for line in stderr_lines[:5]:  # Show first 5 lines
            if line.strip():
                print(f"   {line}")
        if len(result.stderr) > 500:
            print("   ... (truncated)")
    
    print()
    
    summary = {
        'task': task,
        'success_rate': success_cnt / total_cnt if total_cnt > 0 else 0,
        'success_count': success_cnt,
        'total_count': total_cnt,
        'timeout_count': timeout_cnt,
        'avg_steps': total_steps / success_cnt if success_cnt > 0 else 0,
        'total_time': total_time,
        'returncode': result.returncode,
        'run_ids': current_run_ids,
    }
    if task_log_dir is not None:
        summary['log_dir'] = relpath(task_log_dir)
        summary['artifacts_dir'] = relpath(artifacts_dir)
        summary['stdout_log'] = relpath(os.path.join(task_log_dir, "stdout.log"))
        summary['stderr_log'] = relpath(os.path.join(task_log_dir, "stderr.log"))

    write_task_summary(task_log_dir, summary)

    return summary

if __name__ == "__main__":
    import time as time_module

    with terminal_log():
        # tasks = ["sort", "cabinet", "rope", "sweep", "sandwich", "pack"]
        eval_num_runs = int(os.environ.get("ROCO_EVAL_NUM_RUNS", "1"))

        begin_time = time_module.time()

        results = []

        # Use default timeout (from DEFAULT_RUN_TIMEOUTS)
        results.append(test_run_dialog("sort", eval_num_runs, "output"))
        results.append(test_run_dialog("cabinet", eval_num_runs, "output"))
        results.append(test_run_dialog("rope", eval_num_runs, "output"))
        results.append(test_run_dialog("sweep", eval_num_runs, "output"))
        results.append(test_run_dialog("sandwich", eval_num_runs, "output"))
        results.append(test_run_dialog("pack", eval_num_runs, "output"))

        end_time = time_module.time()
        total_elapsed = end_time - begin_time

        print("\n" + "=" * 80)
        print(Colors.BOLD + Colors.CYAN + " " * 30 + "FINAL SUMMARY" + Colors.ENDC)
        print("=" * 80)

        # Header
        print(Colors.BOLD + f"{'Task':<12} {'Success':<12} {'Rate':<10} {'Timeouts':<10} {'Avg Steps':<12} {'Time':<10}" + Colors.ENDC)
        print("─" * 80)

        # Results table
        for result in results:
            task_name = result['task']
            success_str = f"{result['success_count']}/{result['total_count']}"
            rate = result['success_rate'] * 100

            # Color code based on success rate
            if rate >= 80:
                rate_color = Colors.GREEN
            elif rate >= 50:
                rate_color = Colors.YELLOW
            else:
                rate_color = Colors.RED

            rate_str = f"{rate:.1f}%"

            # Color timeouts if any
            if result['timeout_count'] > 0:
                timeout_str = Colors.YELLOW + f"{result['timeout_count']}" + Colors.ENDC
                timeout_padding = " " * (10 - len(str(result['timeout_count'])))
            else:
                timeout_str = f"{result['timeout_count']}"
                timeout_padding = " " * (10 - len(str(result['timeout_count'])))

            avg_steps_str = f"{result['avg_steps']:.2f}" if result['avg_steps'] > 0 else "N/A"
            avg_steps_padding = " " * (12 - len(avg_steps_str))

            time_val = f"{result['total_time']:.1f}s"
            time_str = Colors.BLUE + time_val + Colors.ENDC
            time_padding = " " * (10 - len(time_val))

            print(f"{task_name:<12} "
                  f"{success_str:<12} "
                  f"{rate_color}{rate_str:<10}{Colors.ENDC} "
                  f"{timeout_str}{timeout_padding} "
                  f"{Colors.CYAN}{avg_steps_str}{Colors.ENDC}{avg_steps_padding} "
                  f"{time_str}{time_padding}")

        print("─" * 80)
        print(Colors.BOLD + "Total Execution Time: " + Colors.ENDC + Colors.BLUE + f"{total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)" + Colors.ENDC)
        print("=" * 80 + "\n")

        write_run_summary(results, total_elapsed)
