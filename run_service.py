import subprocess as sub

cmd = ["python", "main_frequentist.py"]
#cmd = ["python", "main_bayesian.py"]
cmd2 = ["./cpu_watt.sh","freq_1_cpu_watts"]
cmd3 = ["./mem_free.sh", "freq_1_ram_use"]

##from time import sleep

path = sub.check_output(['pwd'])
path = path.decode()
path = path.replace('\n', '')

startWattCounter = 'python ' + path + '/gpu_sample_draw.py'

#test = startNODE.split()
#test.append(pythonEnd)
#test.append(pythonEnd2)

#startNODE = test

##print(startNODE)
##print(startWattCounter)

p1 = sub.Popen(cmd)
#p2 = sub.Popen(startWattCounter.split())
p3 = sub.Popen(cmd2)
p4 = sub.Popen(cmd3)

retcode = p1.wait()
print(retcode)

p1.kill()
#p2.kill()
p3.kill()
p4.kill()
