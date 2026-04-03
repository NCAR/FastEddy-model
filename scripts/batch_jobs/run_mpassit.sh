#!/bin/bash

#PBS -S /bin/csh
#PBS -N mpassit
#PBS -A P48503002
#PBS -l walltime=50:00
#PBS -q main
#PBS -o mpassit.out
#PBS -j oe 
#PBS -k oed
#PBS -l select=8:ncpus=32:mpiprocs=32
#PBS -l job_priority=premium
#PBS -m n
#PBS -V

##SBATCH -J mpassit
##SBATCH -o logs/mpassit.%j
##SBATCH -e logs/mpassit.%j
##SBATCH -n 1200
##SBATCH --exclusive
##SBATCH --partition=hera
##SBATCH -t 02:00:00
##SBATCH -A hmtb

#
# This script runs MPASSIT
#

start_init=20240216170000 # YYYYMMDDHHMMSS format
diag_output_interval=300 #interval between lbc files in seconds
FCST_RANGE=5400 #length of forecast in seconds
MPAS_EXPT_DIR=/glade/derecho/scratch/wmayfield/dtc_ncar_mpas/expt_dirs/DTC_NCAR_hrrrIC/conus_3km_fasteddy/mpas_atm/2024021617/ens_1 #directory containing diag, history, init files 
MPASSIT_CODE_DIR=/glade/campaign/ral/jntp/mayfield/fasteddy/MPASSIT #path to base MPASSIT code directory (contains ./bin/mpassit)
VARLIST_DIR=/glade/campaign/ral/jntp/mayfield/fasteddy/run_mpassit/varlists_mpassit_fasteddy #directory containing the variable lists
TOOL_DIR=/glade/u/home/schwartz/utils/derecho #directory with compiled WRFDA tools, in order to use "da_advance_time.exe"

# Load modules:
module --force purge
module use ${MPASSIT_CODE_DIR}/modulefiles
module load build.derecho.intel 

#Start and End dates
DATE=`$TOOL_DIR/da_advance_time.exe ${start_init} 0 -f ccyymmddhhnnss`   # 
end_time=`$TOOL_DIR/da_advance_time.exe ${DATE} ${FCST_RANGE}s -f ccyymmddhhnnss`
echo "Starting init is $start_init"
echo "End time is $end_time"

while [[ "$DATE" -le "$end_time" ]] ; do
    
   # -------------------------------------
   # Get current date into proper format
   # -------------------------------------
   date_file_format=`${TOOL_DIR}/da_advance_time.exe $DATE 0 -f ccyy-mm-dd_hh.nn.ss`
   echo "Date in mpas format is: ${date_file_format}"
   vhr=`echo ${date_file_format}`

   # ----------------------------------
   # Make and go to working directory
   # ----------------------------------
   rundir=./${vhr}
   mkdir -p $rundir
   cd $rundir

   #-----------------------------------------------------------
   # Link necessary input files and code and fill namelist
   #-----------------------------------------------------------
   ln -sf ${MPASSIT_CODE_DIR}/bin/mpassit .
   ln -sf ${VARLIST_DIR}/* .

   export grid_file=${MPAS_EXPT_DIR}/init.nc
   export hist_file=${MPAS_EXPT_DIR}/history.${date_file_format}.nc
   export diag_file=${MPAS_EXPT_DIR}/diag.${date_file_format}.nc
   export output_file=../proc.${date_file_format}.nc

   #--------------------------------------------
   # Create the MPASSIT namelist.input
   #--------------------------------------------

   rm -f ./namelist.input
   cat > ./namelist.input << EOF
&config
grid_file_input_grid="${grid_file}"
hist_file_input_grid="${hist_file}"
diag_file_input_grid="${diag_file}"
file_target_grid="/this/is/an/uneeded/path"
output_file="${output_file}"
target_grid_type = 'lambert'
interp_diag=.true.
interp_hist=.true.
wrf_mod_vars=.true.
esmf_log=.false.
nx = 1578
ny = 925
dx = 3000.0
dy = 3000.0
ref_lat = 38.4
ref_lon = -97.0
truelat1 = 38.4
truelat2 = 38.4
stand_lon = -97.0 /
EOF

   #----------------------------------------------------
   # Run MPASSIT
   #----------------------------------------------------
   rm -f ./*.log
   rm -f ./core*
   rm -f ./*.err

   mpirun ./mpassit namelist.input

   if [[ "$status" -ne "0" ]] ; then
      echo "MPASSIT failed. Exit." >> ./FAIL
      exit 6
   fi

   # Done with this forecast hour; go to next one
   cd ..
   DATE=`$TOOL_DIR/da_advance_time.exe ${DATE} ${diag_output_interval}s -f ccyymmddhhnnss`
   echo "Date is now ${DATE}"
done # loop over time/initializations

exit 0
