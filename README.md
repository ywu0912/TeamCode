# Evolutionary Multiform Optimization with Two-stage Bidirectional Knowledge Transfer Strategy for Point Cloud Registration
by Yue Wu, Hangqi Ding, Maoguo Gong, A. K. Qin, Wenping Ma, Qiguang Miao, Kay Chen Tan, and details are in [paper](https://ieeexplore.ieee.org/abstract/document/9925083).

## Usage
1. Description of running environment: The packaged executable is done with the help of MATLAB Compiler, so the user needs to download Runtime.
   MATLAB Runtime is a set of independent shared libraries, which can run compiled MATLAB applications or components without installing MATLAB.
3. The project folder provided includes 4 folders and 1 file:
  * for_redistribution: the file used to install the application and MATLAB Runtime, the runtime environment under this folder is installed before running the           packaged executable;
  * for_redistribution_files_only: the packaged standalone executables;
  * for_testing: files created by MCC, like binaries and jars, headers and source files, which are used to test the effect of packaging;
  * PackagingLog.html: log files generated by the compiler; .prj: is simply ignored.
3. Run the .exe file on a computer without Matlab installed, and place the folder on a computer without Matlab installed at:
   Run the MyAppInstaller_web.exe file in for_redistribution to install Matlab Runtime;\<br>
   Run the .exe file in for_redistribution_files_only, and you can find that the program runs successfully and shows the point cloud alignment visualization\<br>
   result graph.

## Dataset
* [The Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/)

## Citation
If you find the code or algorithm useful, please consider citing:<br>
   @ARTICLE{9925083,<br>
   author={Wu, Yue and Ding, Hangqi and Gong, Maoguo and Qin, A. K. and Ma, Wenping and Miao, Qiguang and Tan, Kay Chen},<br>
   journal={IEEE Transactions on Evolutionary Computation},<br>
   title={Evolutionary Multiform Optimization with Two-stage Bidirectional Knowledge Transfer Strategy for Point Cloud Registration},<br>
   year={2022},<br>
   pages={1-1},<br>
   doi={10.1109/TEVC.2022.3215743}<br>
   }
