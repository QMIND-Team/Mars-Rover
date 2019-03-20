from keras.models import Model
import params

model = params.model_factory()

with open('./nn_struct.json', 'w') as fout:
    fout.write(model.to_json())
