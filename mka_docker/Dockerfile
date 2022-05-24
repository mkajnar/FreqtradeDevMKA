FROM python:3.9.10 AS build
ENV TZ=Europe/Prague
RUN apt update -y && apt install sudo -y && apt install mc -y
RUN apt install psmisc -y && apt install htop -y
RUN apt install nano -y
RUN     apt-get install openssh-server -y && apt-get install dbus -y
RUN     ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN mkdir /run/sshd && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && echo "root:FreqTrade" | chpasswd
RUN apt update -y
RUN apt install -y python3-pip python3-venv python3-dev python3-pandas git curl
RUN git clone https://github.com/freqtrade/freqtrade.git && cd freqtrade && git checkout develop
WORKDIR /freqtrade
RUN chmod 777 -R /freqtrade
RUN ./setup.sh -i
RUN pip install freqtrade && freqtrade install-ui
EXPOSE 22
EXPOSE 8095
CMD ["/usr/sbin/sshd", "-D"]