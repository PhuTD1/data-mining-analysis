import joblib

model = joblib.load('model/sam_model.pkl')

print(model.predictFromComment('Almost good, but the wheel is so bad, it stopped just 2 months after buying.'))