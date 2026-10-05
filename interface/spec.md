I want to design an interface for users to be able to process new images and explore existing data. It should be very bare bones streamlit app right now.

the pipeline should have tabs where I select either process new or explore data
# process new tab:
    Processing is the process of running inference and postprocessing steps of the pipeline. 
  allow user to specify an image folder (to either recursively use images from or just its own images) or just one image. 
  User should be able to configure the base_config values. 

  preview: 
  While processing, should show a progress bar and a preview of the most recent measurement image with a spinning wheel to show the segmentations and measurements are still being computed. IN in the processing tab, WHile there is no measurment image, the preview of the first (or only)raw  image should appear.  Once it is done, the preview raw image should be replaced with a preview of the measurement image (like test_euler_measurement.jpg.)
 

# explore data tab:
 should pull all results.csv in a selected folder recursively. Default should be C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Raw images
