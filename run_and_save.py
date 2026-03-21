import subprocess
import sys

result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/core/', '-q'], 
                       capture_output=True, text=True)
with open('test_output.log', 'w') as f:
    f.write("STDOUT:\n")
    f.write(result.stdout)
    f.write("\nSTDERR:\n")
    f.write(result.stderr)
    f.write(f"\nReturn code: {result.returncode}")
print("Done")
