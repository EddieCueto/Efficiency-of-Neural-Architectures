import arguments
from time import sleep
import subprocess as sub
from arguments import makeArguments

args = makeArguments(arguments.all_args)

check = list(args.values())
if all(v is None for v in check):
    raise Exception("One argument required")
elif None in check:
    if args['f'] is not None:
        cmd = ["python", "main_frequentist.py"]
    elif args['b'] is not None:
        cmd = ["python", "main_bayesian.py"]
else:
    raise Exception("Only one argument allowed")


wide = args["f"] or args["b"]

with open("tmp", "w") as file:
    file.write(str(wide))

if args['EarlyStopping']:
    with open("stp", "w") as file:
        file.write('2')
elif args['EnergyBound']:
    with open("stp", "w") as file:
        file.write('3')
elif args['AccuracyBound']:
    with open("stp", "w") as file:
        file.write('4')
else:
    with open("stp", "w") as file:
        file.write('1')
        
if args['Save']:
    with open("sav", "w") as file:
        file.write('1')
else:
    with open("sav", "w") as file:
        file.write('0')

sleep(3)


if cmd[1] == "main_frequentist.py":
    cmd2 = ["./cpu_watt.sh", "freq_{}_cpu_watts".format(wide)]
    cmd3 = ["./mem_free.sh", "freq_{}_ram_use".format(wide)]
    with open("frq", "w") as file:
        file.write(str(1))
    with open("bay", "w") as file:
        file.write(str(0))
elif cmd[1] == "main_bayesian.py":
    cmd2 = ["./cpu_watt.sh", "bayes_{}_cpu_watts".format(wide)]
    cmd3 = ["./mem_free.sh", "bayes_{}_ram_use".format(wide)]
    with open("bay", "w") as file:
        file.write(str(1))
    with open("frq", "w") as file:
        file.write(str(0))


path = sub.check_output(['pwd'])
path = path.decode()
path = path.replace('\n', '')

startWattCounter = 'python ' + path + '/amd_sample_draw.py'

#test = startNODE.split()
#test.append(pythonEnd)
#test.append(pythonEnd2)

#startNODE = test

#print(startNODE)
#print(startWattCounter)

p1 = sub.Popen(cmd)
p2 = sub.Popen(startWattCounter.split())
p3 = sub.Popen(cmd2)
p4 = sub.Popen(cmd3)

retcode = p1.wait()
print("Return code: {}".format(retcode))

p1.kill()
p2.kill()
p3.kill()
p4.kill()
