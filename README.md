# Astronomy100-Cruz Allison-HR diagram and aging of M67 Open Cluster
Within this download you should find two different python codes, one named Reduction&Mosaic.py and the other named HR-Diagram&CSV.py


For either of these to work you first need Data to go along with them. This data is stored in a google drive that can be downloaded at this link here


https://drive.google.com/drive/folders/1CvefFN1h1UUQXMcah3WXTow3PUKpvOTG?usp=drive_link


First steps first, download that folder from that google drive. Within it you will find two different kinds of data. The first kind is data collected from Kepler Cam, this includes our two science fits files, as well as all of our flats and biases needed for the reduction pipeline. The other kind is six differently labelled Isochrones that you will be able to use in HR-Diagram&CSV.py 


Once the folder is downloaded I recommend keeping it labelled data for simplicities sake, but in all honesty whatever you want to label it works, as you will need to individually enter the different pathways in the code to make it run. I also recommend adding the data folder to the Astro100CruzProject folder. If you want to fully go through the reduction steps to make sure that this is fully and entirely reproducable feel free to continue with this section, otherwise you can skip to the section labelled Analysis and continue from there. The only caveat with the reduction is that you will have to go to a website online as part of the process. If you're down for some reduction, open up Reduction&Mosaic.py in order to get to work. 


# Reduction


If you are using a python workspace like Cursor that has a built in workspace directory, you may have to drag your data folder into that directory in order to get it kicking, if not however you will just be able to fully enter in the pathway to the files and go from there. Run the python code and you should be prompted for a full pathway to your file directory. Here you will want to enter the full pathway to the either of the files labelled g_band.UR.fits or r_band.UR.fits as these are the un-reduced (UR) science files that you will want to reduce. Once you enter the pathway and wait a few seconds it should export into your data folder a reduced fits file and a reduced and mosaiced fits file labelled "file name"Reduced.fits and 
"file name"Reduced_mosaic.fits respectively. Do this process for both the g_band and r_band fits files.


Now go to Nova.Astrometry.net/upload for the final reduction process. Here click on choose file, find your Reduced_mosaic.fits file and press upload. After waiting for a few minutes a blue link that says Go to results page should appear. Click on that and you'll be taken to a page with a lot of information on it. Here you should be able to, on the right hand side of the page with all of the blue hyper links, find a link labelled new-image.fits. Click on that and it should begin the download process for your now fully reduced fits file. When it downloads, I recommend renaming it to something like r_band_reduced.Fits and moving it to your data folder. Repeat this step for your other reduced_mosaic file and you should now be set to run the HR diagram code!


# Analysis


Now, open HR-Diagram&CSV.py. This step is a little more involved than the reduction step. Within the code, below all of the initial imports you should be able to find a clearly labelled section that asks you to input the path directory for your reduced g-band and r-band files. If you skipped the reduction step this should just be labelled as r_band_preduced.fits and g_band_preduced.fits (the pre-reduced files ;).) From here you will also have to scroll down to where it asks you to enter the path to the .dat isochrone file. This should work as written if you are running the code in cursor and the file is in your data folder within your cursor directory (which it should be if the data folder is within the Astro100 folder. If this doesn't work, go in front of the quotation marks, add an r [like r""] and you should be able to directly enter the full pathway to the isochrone file.) Optionally you can also enter a path directory to save a plot of the apertures. Now all you need to press is run and it should print out an HR diagram with the isochrone overlayed on top of it. It will also export a csv file, which is a star table that it generates. It does this because personally I found working with my data in glue to be incredibly convenient, but its not at all necessary. In order to change which isochrone you're looking at all you have to do is go to that isochrone section in the code and change the file to whichever one you want. Now it should all be done! 
