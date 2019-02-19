# Run in Docker Container

* Clone the repo 
* unzip the creditcard.csv.zip -> creditcard.csv
* Build the docker container "docker build -t mlprac1 ." 
* Run the docker container "docker run -p 8888:8888 mlprac1:latest"
* Paste the link into your browser (http://localhost:8888/?token=xxxxxxx on a windows machine)