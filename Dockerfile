FROM python:3.8

# set a directory for the app
WORKDIR /opt/feature-extractor

# copy all the files to the container
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# run the command
CMD ["sh", "./example.sh"]
