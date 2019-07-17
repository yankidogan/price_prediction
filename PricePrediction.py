
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

dataset = pd.read_csv('AutoDataset.csv')

change_gearbox = dataset['gearbox'].apply(str).str.get_dummies().add_prefix('Gearbox: ')
change_fuelType = dataset['fuelType'].apply(str).str.get_dummies().add_prefix('FuelType: ')
change_brand = dataset['brand'].apply(str).str.get_dummies().add_prefix('Brand: ')
change_notRepairedDamage = dataset['notRepairedDamage'].apply(str).str.get_dummies().add_prefix('Damage: ')

dataset = pd.concat([dataset, change_gearbox, change_fuelType, change_brand, change_notRepairedDamage], axis=1)
dataset = dataset.drop(['gearbox', 'fuelType', 'brand', 'notRepairedDamage'], axis=1)

X = dataset[['yearOfRegistration',
        'powerPS',
        'kilometer',
        'Gearbox: automatik',
        'Gearbox: manuell',
        'FuelType: benzin',
        'FuelType: diesel',
        'FuelType: lpg',
        'Brand: audi',
        'Brand: bmw',
        'Brand: ford',
        'Brand: honda',
        'Brand: mercedes_benz',
        'Brand: opel',
        'Brand: peugeot',
        'Brand: porsche',
        'Brand: renault',
        'Brand: volkswagen',
        'Damage: ja',
        'Damage: nein']]
y = dataset['price'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

X_normalizer = StandardScaler()
X_train = X_normalizer.fit_transform(X_train)
X_test = X_normalizer.transform(X_test)

y_normalizer = StandardScaler()
y_train = y_normalizer.fit_transform(y_train)
y_test = y_normalizer.transform(y_test)

model = MLPRegressor(hidden_layer_sizes=(100, 100))
model.fit(X_train, y_train.ravel())



button_predict = widgets.Button(description="Predict")

checkbox_damage = widgets.Checkbox(value=False, description='Is damaged', disabled=False)

dropdown_brand = widgets.Dropdown(
    options=['Audi', 'BMW', 'Ford', 'Honda', 'Mercedes Benz', 'Opel', 'Peugeot', 'Porsche', 'Renault', 'Volkswagen'],
    value='Audi',
    description='Brand:',
    disabled=False)

text_yor = widgets.IntText(value=0,
    description='Year of Registration:',
    disabled=False
)

text_km = widgets.IntText(value=0,
    description='Kilometer:',
    disabled=False
)

text_power = widgets.IntText(value=0,
    description='Power (PS):',
    disabled=False
)

toggle_gear = widgets.ToggleButtons(
    options=['Automatic', 'Manual'],
    description='Gearbox:',
    disabled=False,
    button_style='',
)

toggle_fuel = widgets.ToggleButtons(
    options=['Petrol', 'Diesel', 'LPG'],
    description='Fuel Type:',
    disabled=False,
    button_style='',
)


display(dropdown_brand)
display(text_yor)
display(text_km)
display(text_power)
display(checkbox_damage)
display(toggle_fuel)
display(toggle_gear)
display(button_predict)

def on_button_clicked(b):
    yor, power, km, gearb_a, gearb_m, fuel_b, fuel_d, fuel_l, brand_a, brand_b, brand_f, brand_h, brand_m, brand_o, brand_p, brand_po, brand_r, brand_v, damage_j, damage_n = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    if checkbox_damage.value:
        damage_j=1
    
    else:
        damage_n=1
    
    Brand = dropdown_brand.value
    if Brand=='Audi':
        brand_a=1
    elif Brand=='BMW':
        brand_b=1
    elif Brand=='Ford':
        brand_f=1
    elif Brand=='Honda':
        brand_h=1
    elif Brand=='Mercedes Benz':
        brand_m=1
    elif Brand=='Opel':
        brand_o=1
    elif Brand=='Peugeot':
        brand_p=1
    elif Brand=='Porsche':
        brand_po=1
    elif Brand=='Renault':
        brand_r=1
    else:
        brand_v=1
        
    Gearbox = toggle_gear.value
    if Gearbox=='Automatic':
        gearb_a=1
    else:
        gearb_m=1
        
    FuelType = toggle_fuel.value
    if FuelType=='Petrol':
        fuel_b=1
    elif FuelType=='Diesel':
        fuel_d=1
    else:
        fuel_l=1
        
    yor=text_yor.value
    power=text_power.value
    km=text_km.value
    
    cardetails = pd.DataFrame([
    {
        'yearOfRegistration': yor,
        'powerPS': power,
        'kilometer': km,
        'Gearbox: automatik': gearb_a,
        'Gearbox: manuell': gearb_m,
        'FuelType: benzin': fuel_b,
        'FuelType: diesel': fuel_d,
        'FuelType: lpg': fuel_l,
        'Brand: audi': brand_a,
        'Brand: bmw': brand_b,
        'Brand: ford': brand_f,
        'Brand: honda': brand_h,
        'Brand: mercedes_benz': brand_m,
        'Brand: opel': brand_o,
        'Brand: peugeot': brand_p,
        'Brand: porsche': brand_po,
        'Brand: renault': brand_r,
        'Brand: volkswagen': brand_v,
        'Damage: ja': damage_j,
        'Damage: nein': damage_n
        }
        ])

    customx = cardetails[['yearOfRegistration',
        'powerPS',
        'kilometer',
        'Gearbox: automatik',
        'Gearbox: manuell',
        'FuelType: benzin',
        'FuelType: diesel',
        'FuelType: lpg',
        'Brand: audi',
        'Brand: bmw',
        'Brand: ford',
        'Brand: honda',
        'Brand: mercedes_benz',
        'Brand: opel',
        'Brand: peugeot',
        'Brand: porsche',
        'Brand: renault',
        'Brand: volkswagen',
        'Damage: ja',
        'Damage: nein']]
    customx = X_normalizer.transform(customx)

    pred = model.predict(customx)
    prediction = y_normalizer.inverse_transform(pred)
    print('Predicted price: €%.2f' % prediction)
    

button_predict.on_click(on_button_clicked)




