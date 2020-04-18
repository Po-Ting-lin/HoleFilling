# HoleFilling

## Table of Contents
* Introduction
* testing result

### Introduction
Implementation of [Improving Morphology Operation for 2D Hole Filling Algorithm](http://www.cscjournals.org/library/manuscriptinfo.php?mc=IJIP-493)
.This paper proposed a faster solution than the original 2D hole filling algorithm by applying a pre-marker mask to initiate the process of hole filling.
It greatly reduces computational process on sparse images.  


### testing result
* test image  
![](/sample/hole.png)  

* result  
1. Border Image Initial Algorithm  
![](/sample/BIIA.png)  
time consuming: 1.5 s  


2. Speed Border Image Initial Algorithm  
![](/sample/result.png)  
time consuming: 0.2 s  


