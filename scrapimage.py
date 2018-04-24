from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"Dhaulagiri","limit":100, "output_directory":"pics","print_urls":True}   #creating list of arguments
response.download(arguments)
