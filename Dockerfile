# Use an appropriate base image with Python support and Temurin JDK 11
FROM python:3.10

# Install system dependencies required for PyLucene and jcc
RUN apt-get update

# Set the JCC_JDK environment variable to the correct Java JDK directory
ENV JCC_JDK=/usr/lib/jvm/temurin-17-jdk-amd64
RUN mkdir -p $JCC_JDK 
RUN curl -L https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.8%2B7/OpenJDK17U-jdk_x64_linux_hotspot_17.0.8_7.tar.gz | tar xz -C $JCC_JDK --strip-components=1
ENV PATH=$JCC_JDK/bin:$PATH

# Download PyLucene
# Install PyLucene
# Commenting out the lines that download and extract PyLucene
# ARG PYLUCENE_VERSION=9.7.0
# RUN wget -q https://dlcdn.apache.org/lucene/pylucene/pylucene-${PYLUCENE_VERSION}-src.tar.gz
# RUN tar -xzvf pylucene-${PYLUCENE_VERSION}-src.tar.gz
# RUN mv pylucene-${PYLUCENE_VERSION} /usr/src/pylucene-src
# Instead, copy a local directory pylucene-8.11.0-src to /usr/src/pylucene-src
COPY pylucene-9.7.0 /usr/src/pylucene-src

# Install jcc
WORKDIR /usr/src/pylucene-src/jcc
RUN python3 setup.py build
RUN python3 setup.py install

# Install icupkg, which is needed by PyLucene make
RUN apt-get install libicu-dev

# Install PyLucene
WORKDIR /usr/src/pylucene-src
RUN make clean
RUN make
RUN make install

# Install Sparkly
RUN apt-get install -y git
WORKDIR /usr/src
RUN git clone https://github.com/anhaidgroup/sparkly.git
WORKDIR /usr/src/sparkly
RUN python3 -m pip install -r ./requirements.txt

# Copy the shell setup and python script to the container
# COPY gridsearch.py /usr/src/sparkly
# COPY gridsearch.sh /usr/src/sparkly
# COPY extra_requirements.txt /usr/src/sparkly

# # Mark the shell script as executable
# RUN chmod +x entrypoint.sh

# # Set the shell script as the entrypoint
# ENTRYPOINT ["./entrypoint.sh"]
