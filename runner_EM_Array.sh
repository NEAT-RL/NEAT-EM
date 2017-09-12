#!/bin/sh
#
# Force Bourne Shell if not Sun Grid Engine default shell (you never know!)
#
#$ -S /bin/sh
#
# I know I have a directory here so I'll use it as my initial working directory
#
#$ -wd /vol/grid-solar/sgeusers/singhharm1/NEATEM_THEANO
#
# Mail me at the b(eginning) and e(nd) of the job
#$ -M Harman.Singh@ecs.vuw.ac.nz
#$ -M be
#$ -notify
# End of the setup directives
#
# Now let's do something useful, but first change into the job-specific
# directory that should have been created for us
#
# Check we have somewhere to work now and if we don't, exit nicely.
#

DIRECTORY="NEATEM_THEANO"

if [ -d /local/tmp/singhharm1/$JOB_ID.$SGE_TASK_ID ]; then
        cd /local/tmp/singhharm1/$JOB_ID.$SGE_TASK_ID
else
        echo "Uh oh ! There's no job directory to change into "
        echo "Something is broken. I should inform the programmers"
        echo "Save some information that may be of use to them"
        echo "Here's LOCAL TMP "
        ls -la /local/tmp
        echo "AND LOCAL TMP singhharm1 "
        ls -la /local/tmp/singhharm1
        echo "Exiting"
        exit 1
fi
#
# Now we are in the job-specific directory so now can do something useful
#
# Stdout from programs and shell echos will go into the file
#    scriptname.o$JOB_ID
#  so we'll put a few things in there to help us see what went on
#
#
# Do specific stuff here. Here i need to first use bash, then conda env list, source activate openai-neat.
#
echo ==UNAME==
uname -n
echo ==WHO AM I and GROUPS==
id
groups
echo ==SGE_O_WORKDIR==
echo $SGE_O_WORKDIR
echo ==/LOCAL/TMP==
ls -ltr /local/tmp/
echo ==/VOL/GRID-SOLAR==
ls -l /vol/grid-solar/sgeusers/
#
# OK, where are we starting from and what's the environment we're in
#
echo ==RUN HOME==
pwd
ls
echo ==ENV==
env
echo ==SET==
set
#
echo == WHATS IN LOCAL/TMP ON THE MACHINE WE ARE RUNNING ON ==
ls -ltra /local/tmp | tail
#
echo == WHATS IN LOCAL TMP singhharm1 JOB_ID AT THE START==
ls -la

#
# Clone repo
#
echo ==CLONE REPO==
pwd
rm -r -f NEAT-EM
git clone git@github.com:NEAT-RL/NEAT-EM.git
wait

echo ==WHATS THERE HAVING CLONED STUFF==
ls -la

#
# Run python environment in bash
#
echo ==SETUP BASH==
bash
export PATH=/home/singhharm1/miniconda3/bin:$PATH

echo ==SETUP CONDA ENV==
conda env list
source activate openai-neat

#
# cd into repo
#
echo ==GOING INTO EM DIRECTORY==
cd NEAT-EM

#
# Run alrogithm
#
echo ==RUNNING ALGORITHM==
python NEATEM.py --threads 0
wait

#
# Now we move the output to a place to pick it up from later
#  (really should check that directory exists too, but this is just a test)
#
echo ==COPY PROGRAM RUN FILES==
mkdir -p /vol/grid-solar/sgeusers/singhharm1/$DIRECTORY/$JOB_ID/$SGE_TASK_ID
cp agent*  /vol/grid-solar/sgeusers/singhharm1/$DIRECTORY/$JOB_ID/$SGE_TASK_ID
cp log/debug* /vol/grid-solar/sgeusers/singhharm1/$DIRECTORY/$JOB_ID/$SGE_TASK_ID
# cp output.log /vol/grid-solar/sgeusers/singhharm1/$DIRECTORY/$JOB_ID/$SGE_TASK_ID

#
echo "Ran through OK"
