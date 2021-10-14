import requests
import matplotlib.pyplot as plt
import io
import PIL
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
import os
import apscheduler
import bs4
from apscheduler.schedulers.blocking import BlockingScheduler
import warnings
warnings.filterwarnings("ignore")

API_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' #change here
TOP_LATITUDE = 52.993549134817094
TOP_LONGTITUDE = -1.2540469137135828
BOTTOM_LATITUDE = 52.93198796565878
BOTTOM_LONGTITUDE = -1.0569210510523759
CSV_DATA_PATH = 'csv_data'
IMAGE_DATA_PATH = 'img_data'
sequence = 0

def get_xlm_data():
    request_string = 'https://traffic.ls.hereapi.com/traffic/6.2/flow.xml?apiKey={}&bbox={},{};{},{}&responseattributes=sh,fc'.format(API_KEY, TOP_LATITUDE, TOP_LONGTITUDE, BOTTOM_LATITUDE, BOTTOM_LONGTITUDE)
    xlm = requests.get(request_string)

    return xlm

def get_traffic_flow_data(xlm_data):
    '''
    #DE: street name
    #FF: free flow speed
    #JF: Jam Factor
    #SP: Speed capped
    #SU: Speed uncapped
    #QD: Direction (+/-)
    #CN: Confidence Level
    '''
    
    tree = ET.fromstring(xlm_data.content)
    timestamp = tree[0][0].attrib.get('PBT')

    data_vectors = []
    data_vectors.append(timestamp)
    headings = ['Street', 'District', 'Direction', 'FreeFlow', 'SpeedUncapped', 'SpeedCapped', 'JamFactor']
    data_vectors.append(headings)

    for i in range(len(tree[0])):
        # print('------- new section ---------')
        # print(tree[0][i].attrib)

        for d in tree[0][i]:
            street = tree[0][i].attrib.get('DE')
            district = direction = d[0][0].attrib.get('DE')
            direction = d[0][0].attrib.get('QD')
            free_flow = d[0][-1].attrib.get('FF')
            speed_uncapped = d[0][-1].attrib.get('SU')
            speed_capped = d[0][-1].attrib.get('SP')
            jam_factor = d[0][-1].attrib.get('JF')

            # print(d[0][0].attrib)
            # print(d[0][-1].attrib)
            data_vectors.append([street, district, direction, free_flow, speed_uncapped, speed_capped, jam_factor])

    return data_vectors, timestamp

def save_data(data_vectors):
    df=pd.DataFrame(data_vectors)
    df.to_csv(CSV_DATA_PATH + '/' + 'traffic_flow.csv', mode='a', header=False, index=False)

def get_roads(xlm_data):
    soup = bs4.BeautifulSoup(xlm_data.text, "html.parser")
    roads = soup.find_all('fi')
    return roads

def save_image(roads_x, roads_y, speed_factor, timestamp):
    fig=plt.figure()
    plt.style.use('dark_background')
    plt.grid(False)
    for i in range(0,len(roads_x)):
        sf = speed_factor[i]
        if(sf < 0.25):
            plt.plot(roads_x[i],roads_y[i], c='red',linewidth=1.5)
        elif(sf < 0.5):
            plt.plot(roads_x[i],roads_y[i], c='orange',linewidth=1.5)
        elif(sf < 0.75):
            plt.plot(roads_x[i],roads_y[i], c='yellow',linewidth=1.5)
        else:
            plt.plot(roads_x[i],roads_y[i], c='green',linewidth=1.5)

    plt.axis('off')
    
    
    path, dirs, files = next(os.walk(IMAGE_DATA_PATH))
    file_count = len(files)
    i = file_count
    filename = 'img_' + str(i) + '.png'
    plt.savefig(fname = IMAGE_DATA_PATH + '/' + filename, facecolor='black', edgecolor='w')
    
def get_image_data(xlm_data):
    roads_info = get_roads(xlm_data)
    road_class = 6
    roads_x = []
    roads_y = []
    speed_factor = []

    for road in roads_info:
        road_xml = ET.fromstring(str(road))

        for child in road_xml:
            end_nodes = child.attrib

            if('fc' in end_nodes):
                fc = int(end_nodes['fc'])
            if('su' in end_nodes):
                su = float(end_nodes['su'])
            if('ff' in end_nodes):
                ff = float(end_nodes['ff'])

        if(fc <= road_class):
            shapes = road.find_all("shp")
            
            for i in range(0, len(shapes)):
                lat_long_pairs = shapes[i].text.replace(',',' ').split()
                x = []
                y = []
                speed_uncapped = []
                free_flow = []
                
                for ii in range(0, int(len(lat_long_pairs)/2)):
                    x.append(float(lat_long_pairs[2*ii]))
                    y.append(float(lat_long_pairs[2*ii+1]))
                    speed_uncapped.append(float(su))
                    free_flow.append(float(ff))

                roads_x.append(x)
                roads_y.append(y)
                speed_factor.append(np.mean(speed_uncapped) / np.mean(free_flow))

    return roads_x, roads_y, speed_factor
    
def execute_loop():

    xlm_data = get_xlm_data()
    traffic_data, timestamp = get_traffic_flow_data(xlm_data)
    x, y, speed_factor = get_image_data(xlm_data)

    save_data(traffic_data)
    save_image(x, y, speed_factor, timestamp)
    print('saved')

def schedule(time):
    scheduler = BlockingScheduler()
    scheduler.add_job(execute_loop, 'interval', hours=time)
    scheduler.start()

def main():
    os.makedirs(CSV_DATA_PATH, exist_ok=True)
    os.makedirs(IMAGE_DATA_PATH, exist_ok=True)
    schedule(time = 0.15)

main()