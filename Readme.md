dashSVD: A Faster Randomized SVD Algorithm with Dynamic Shifts for Sparse Data
---
Paper: Xu Feng, Wenjian Yu, Yuyang Xie, and Jie Tang, “Algorithm xxx: Faster randomized SVD with dynamic shifts,” ACM Transactions on Mathematical Software, 2024.
### 1.Main Algorithms

1.matlab/example/basic_rSVD_shift.m ---- basic randomized SVD algorithm with shifted power iteration and shift updating scheme (Alg. 2 without accelerating skills).

2.matlab/src/dashSVD.m ---- dashSVD with shifted power iteration for sparse data in Matlab (dashSVD with accelerating skills).

3.mkl/src/dashsvd.c ---- dashSVD algorithm in our paper which is implemented in C with MKL and OpenMP.

### 2.Experiments for Testing

(1)The program for testing dashSVD with MKL is in "mkl/example". The MKL library needs the support of Intel MKL [1]. When all the libraries have been prepared, firstly modified the path of MKL in makefile, and secondly use "make" to produce the executable program "dashsvdtest". The result of program is an example of testing the dataset SNAP [2] on dashSVD with and without accuracy control.

(2)matlab/example/ShiftedPowerIteration_Test.m is used to test the effectiveness of shifted power iteration. The comparison is between Alg. 1 (basic_rSVD.m)  [3], Alg. 2* (basic_rSVD_shift_noupdate.m) and Alg. 2 (basic_rSVD_shift.m) on Dense1 or Dense2.

(3)matlab/example/dashSVD_Test.m is used to test the effectiveness of dashSVD compared with LanczosBD in svds in Matlab on SNAP. (Notice that svdstest.m and LanczosBDtest.m is the modification of svds to make sure LanczosBD can produce results with settled times of restarting.)

(4)matlab/example/AccuracyControl_Test.m is used to validate the PVE criterion of accuracy control used in dashSVD.

(5)The matrix index should begin with 1 when using those programs, or the results may become false. Besides, the singular values computed by eigSVD are in ascending order.

### Reference

[1]  Intel oneAPI Math Kernel Library. https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html, 2021. 

[2] Jure Leskovec and Andrej Krevl. SNAP Datasets: Stanford large network dataset collection. http://snap.stanford.edu/data, June 2014.

[3] N Halko, P. G Martinsson, and J. A Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. Siam Review, 53(2):217-288, 2011. 
