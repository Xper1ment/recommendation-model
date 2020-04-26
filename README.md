## recommendation-model

Data is in csv format in data folder for db.

To run install all dependencies from requirement.txt, then run server app.py

# API
The api takes only the film names from url as input. It will send the name and id of the recommended films in json format, which the api decides
by director name,keyword and genre of the film saved in the csv file.
So the node server should send the film name the user wants at /recommend/ url, see the code for further details.


