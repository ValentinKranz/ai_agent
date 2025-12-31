from functions.run_python_file import run_python_file

#should print the calculator's usage instructions
print(run_python_file("calculator", "main.py"))
#should run the calculator... which gives a kinda nasty rendered result
print(run_python_file("calculator", "main.py", ["3 + 5"]))
#should run the calculator's tests successfully
print(run_python_file("calculator", "tests.py"))
#this should return an error
print(run_python_file("calculator", "../main.py"))
#this should return an error
print(run_python_file("calculator", "nonexistent.py"))
#this should return an error
print(run_python_file("calculator", "lorem.txt"))