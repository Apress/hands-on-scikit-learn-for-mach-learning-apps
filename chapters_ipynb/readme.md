**********************************
Notes (Jupyter Notebook IDE users)
**********************************

1. All code tested (November 27, 2019) using Jupyter Notebook Python IDE

2. We noticed that when loading (numpy.load) banking data, need to include parameter
 'allow_pickle=True' when using numpy.load to load target variable 'y' data. Also, need to
 include the same parameter when loading variable 'bp' (best parameters from tuning). The 
code you download from the book website should work without any modifications.

3. If you have any difficulties with numpy.load, you can experiment with the 'allow_pickle' 
parameter. With all of our testing, we have only had to add 'allow_pickle=True' to the 
numpy.load() function. We believe that earlier versions of Python were less strict because 
the code was tested by several people a few months ago with no problems.

4. With notebook files that contain matplotlib images, you may have to run the code twice
 in Jupyter Notebook to see the image displayed.