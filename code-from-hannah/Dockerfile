FROM ubuntu:latest

RUN apt-get update
RUN apt-get -y install automake build-essential clang cmake git libboost-dev libboost-thread-dev libntl-dev libsodium-dev libssl-dev libtool m4 python3 texinfo yasm libgmp-dev iproute2 vim python3-pip
RUN pip3 install numpy

WORKDIR /benchmarks
COPY ./benchmarks .

RUN chmod u+x ./setup_mpspdz.sh && ./setup_mpspdz.sh

RUN echo 'Good luck!'