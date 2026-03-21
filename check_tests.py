import subprocess
import sys

result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/core/', '-q'], 
                       capture_output=True, text=True)
print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
print("Return code:", result.returncode)
