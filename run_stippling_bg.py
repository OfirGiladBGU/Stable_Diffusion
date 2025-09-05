import subprocess
import platform
import os


def launch_stippling():
    log_file = "fast_stippling_generator.log"
    script_to_run = "fast_stippling_generator.py"
    conda_env = "sd"

    # Add timestamp to log start
    with open(log_file, "w") as log:  # Use 'w' to start fresh
        log.write(f"==== Script started at {subprocess.check_output(['date'], text=True).strip()} ====\n")
        log.flush()

    # Use bash to properly source conda and activate environment
    bash_cmd = f"""
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate {conda_env}
python -u {script_to_run}
"""
    
    print(f"\n✅ Launching '{script_to_run}' in background using conda env '{conda_env}'.")
    print(f"📁 Output is being logged to: {log_file}")
    print(f"🔧 Using bash with conda activation")
    
    # Start process in background with output redirection
    with open(log_file, "a") as log:
        process = subprocess.Popen(
            ["bash", "-c", bash_cmd],
            stdout=log,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            start_new_session=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1")
        )

        pid = process.pid
        print(f"🆔 Process ID (PID): {pid}")

        system = platform.system()
        if system == "Windows":
            print(f"🛑 To stop it, run: taskkill /PID {pid} /F")
        else:
            print(f"🛑 To stop it, run: kill {pid}")
            print(f"📊 To monitor logs, run: tail -f {log_file}")


if __name__ == "__main__":
    launch_stippling()
