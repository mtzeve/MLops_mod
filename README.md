For this module we decided to work with stocks prediction combinng it with news/articles and tried to perform a sentiment analysis on top. For that reason we implemented vscode to write our code and hopsworks to store our feature groups, feature view and the model. The APIs that we used are the Alpha vantage for the stocks and the Polygon for the news. 

We decided that what we wanted was to predict the 'open' price for TSLA stocks. This was because when you are buying stocks you can't buy them after the stock market has closed.

Feature group pipeline creates feature groups for the stocks and for the news sentiment. These feature groups were 'merged' when creating the freature view. After the feature view was created the train, test split by using set dates. We decided on a 2 year span because there was a 2 year historical limit with Polygon's news API.

Unfortunately when creating the inference pipeline we ran into issues when using hopsworks get.batch function, which set us back alot, therefore we couldnt finish exactly how we wanted.

