FROM freqtradeorg/freqtrade:develop AS build
RUN pip install requests-html
EXPOSE 8080
ENTRYPOINT ["freqtrade"]
CMD [ "trade" ]
