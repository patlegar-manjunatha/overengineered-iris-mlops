# overengineered-iris-mlops
A production-grade MLOps pipeline built on a toy dataset. This project demonstrates end-to-end lifecycle management (Docker, K8s, CI/CD, MLflow, AWS) applied to the Iris dataset to showcase infrastructure architecture without the distraction of complex modeling.


TODO : 
> dump model & scaler to dagshub S3 after training the model using 
```
dagshub upload --bucket patlegar-manjunatha/overengineered-iris-mlops "artifacts"
```

> load model & scaler from dagshub s3 while prediciton using : 
```
dagshub download --bucket patlegar-manjunatha/overengineered-iris-mlops .
```


