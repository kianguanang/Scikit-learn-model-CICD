# algotuno_skikitlearn_backend
Algotuno Scikit-Learn model (MODEL 2) backend V1
This is the Scikit-Learn backend api using random forest regression model.
This model is hosted on AWS Lambda as a lambda function

## calling the lambda function
send a GET or POST request to the lambda endpoint with the body contents as such: NOTE: the stock_metadata_list should be prefilled with the stock data coming from the /api/stock/get_hsp api

## URL ENDPOINTS TO HIT

>https://q6p47mowxp5fy2dkeh6cvg6dwi0dykpt.lambda-url.us-east-1.on.aws/

Input Example:
```
{
    "ticker_symbol" : "TSLA",
    "stock_metadata_list" : []
}
```
## Output Example:
```
{
    "message": "SUCCESS",
    "result": {
        "ticker_symbol": "APPL",
        "model_type": 1,
        "prediction": [
            {
                "Sat Jan 01 2022 08:00:00 GMT+0800 (Singapore Standard Time)": 177.87269592285156
            },
            {
                "Fri Jan 07 2022 08:00:00 GMT+0800 (Singapore Standard Time)": 178.38754272460938
            },
            {
                "Sun Jan 30 2022 08:00:00 GMT+0800 (Singapore Standard Time)": 178.56460571289062
            }
        ]
    }
}
```

Example with POSTMAN:
![image](https://user-images.githubusercontent.com/103578433/174420756-46d0c70a-92ec-4368-8314-218e0fa1b658.png)
