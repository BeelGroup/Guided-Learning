@ECHO OFF & SETLOCAL
SET /P np=Number of processes:
FOR /L %%I IN (1, 1, %np%) DO (
   START python main.py --name %%I
)
PAUSE