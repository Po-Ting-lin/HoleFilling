# HoleFilling

### Introduction
Implementation of hole filling with Speed Border Image Initial Algorithm[1]. The process is suitible for accelerating by gpu. In the test image(1.3 megapixels), the gain of performance is up to 316 times compared with the CPU Naive method.

### Test   
![](/sample/letter_image-1.png)
![](/sample/letter_result-1.png)  
![](/sample/rbc-1.png)
![](/sample/rbc_result-1.png)  

Method        |Image                 | Avg. Elapsed Time
--------------|:--------------------:|-------------------
CPU Naive     |Text (1.3 Megapixels) |  6322 ms
GPU           |Text (1.3 Megapixels) |  20 ms (~ x316)
GPU           |RBC  (1.4 Megapixels) |  69 ms

### Reference
```
[1] Hasan, Mokhtar M., and Pramod K. Mishra. 
    "Improving morphology operation for 2D hole filling algorithm."
    International Journal of Image Processing (IJIP) 6.1 (2012): 635-646.
```

