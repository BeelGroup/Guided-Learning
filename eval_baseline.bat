@ECHO OFF & SETLOCAL
CD PATH_TO_SCRIPT
SET /P INPUT=Number of processes:
FOR /L %%I IN (1, 1, 2) DO (
   START python main.py --name %%I
)
PAUSE