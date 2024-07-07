import configparser
from itertools import combinations_with_replacement, zip_longest
from matplotlib.backends.backend_pdf import PdfPages
import os
from typing import List

from matplotlib import pyplot as plt
import numpy as np

import pysindy as ps
from pysindy import SINDyDerivative


from sha_learning.case_studies.energy_made.sul_functions import label_event, parse_data, get_power_param, is_chg_pt
from sha_learning.domain.lshafeatures import Event, NormalDistribution, Trace
from sha_learning.domain.sigfeatures import Timestamp, SampledSignal
from sha_learning.domain.sulfeatures import SystemUnderLearning, RealValuedVar, FlowCondition
from sha_learning.learning_setup.teacher import Teacher
from sha_learning.pltr.energy_made_pltr import double_plot

config = configparser.ConfigParser()  # open the configuration file
config.sections()
config.read('./resources/config/config.ini')
config.sections()

SPEED_RANGE = int(config['ENERGY CS']['SPEED_RANGE'])  # get all the constant values from the config file
MIN_SPEED = int(config['ENERGY CS']['MIN_SPEED'])
MAX_SPEED = int(config['ENERGY CS']['MAX_SPEED'])


def pwr_model(interval: List[Timestamp], P_0):
    interval = [ts.to_secs() for ts in interval]
    AVG_PW = 1.0
    return [AVG_PW] * len(interval)


# define flow conditions
on_fc: FlowCondition = FlowCondition(0, pwr_model)

# define distributions
off_distr = NormalDistribution(0, 0.0, 0.0)

model2distr = {0: [], 1: [], 2: []}
power = RealValuedVar([on_fc], [], model2distr, label='P')
speed = RealValuedVar([on_fc], [], model2distr, label='w')

# define events as different velocities ranges and stop, load and unload
events: List[Event] = []
for i in range(MIN_SPEED, MAX_SPEED, SPEED_RANGE):
    if i < MAX_SPEED - SPEED_RANGE:
        new_guard = '{}<=w<{}'.format(i, i + SPEED_RANGE)
    else:
        new_guard = '{}<=w'.format(i)
    events.append(Event(new_guard, 'start', 'm_{}'.format(len(events))))

spindle_off = Event('', 'stop', 'i_0')

events.append(spindle_off)
events.append(Event('', 'load', 'l'))
events.append(Event('', 'unload', 'u'))

DRIVER_SIG = ['w', 'pr', 'id', 'wd']
DEFAULT_M = 0
DEFAULT_DISTR = 0

args = {'name': 'energy', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}
energy_made_cs = SystemUnderLearning([power, speed], events, parse_data, label_event, get_power_param, is_chg_pt, args=args)

''' PySindy Flow Conditions'''
base_path="/home/simo/WebFarm/lsha/resources/traces/MADE/"
data_paths = [base_path+"_03_mar_1.csv",
                base_path+"_05_may_1.csv",
                base_path+"_05_may_2.csv",
                base_path+"_11_jan_2.csv",
                base_path+"_12_apr_1.csv",
                base_path+"_12_apr_2.csv",
                base_path+"_12_jan_2.csv",
                base_path+"_12_jan_3.csv",
                base_path+"_12_jan_4.csv",
                base_path+"_12_jan_5.csv",
                base_path+"_13_feb_1.csv",
                base_path+"_13_feb_2.csv",
                base_path+"_15_feb_1.csv",
                base_path+"_17_feb_2.csv",
                base_path+"_17_feb_3.csv"
                ]
def extractTimestamps(points):
  return [str(point.timestamp).split(' ', 1)[1] for point in points]

def transform_times_to_seconds_cumulative(times):
    # Converte i tempi nel formato 'HH:MM:SS' in secondi totali
    times_seconds = [sum(int(x) * 60**i for i, x in enumerate(reversed(time.split(':')))) for time in times]
    # Calcola il tempo cumulativo trascorso dal primo elemento
    times_transformed = [time - times_seconds[0] for time in times_seconds]
    return np.array(times_transformed)
def generateData(data_path):
  new_signals: List[SampledSignal] = parse_data(data_path)
  chg_pts = energy_made_cs.find_chg_pts([sig for sig in new_signals if sig.label in DRIVER_SIG])
  power_pts = new_signals[0].points
  speed_pts = new_signals[1].points
  pressure_pts = new_signals[2].points
  power_values = [pt.value for pt in power_pts]
  speed_values = [st.value for st in speed_pts]
  id_events = [label_event(events, new_signals, pt.t) for pt in chg_pts[:10]]
  energy_made_cs.process_data(data_path)
  trace = energy_made_cs.timed_traces[-1]

  power_data = np.array([pt.value for pt in power_pts]).ravel()
  speed_data = np.array([pt.value/1000 for pt in speed_pts]).ravel() 
  t_power = transform_times_to_seconds_cumulative(np.array(extractTimestamps(power_pts)))
  t_speed = transform_times_to_seconds_cumulative(np.array(extractTimestamps(speed_pts)))
  return power_data, speed_data, t_power
power_datas = []
speed_datas = []
ts = []

for dp in data_paths:
  pd, sd, t = generateData(dp)
  power_datas.append(pd)
  speed_datas.append(sd)
  ts.append(t)
print(power_datas[0].shape)
train_speed = []
train_power = []
test_speed = []
test_power = []

for i in range(0,len(speed_datas)):
  if i == 1 or i == 2 or i == 4 or i == 5 or i == 7:
    test_speed.append(speed_datas[i])
    test_power.append(power_datas[i])
  else:
    train_speed.append(speed_datas[i])
    train_power.append(power_datas[i])
combined_speed_data = []
combined_power_data = []
for st,pt in zip(train_speed, train_power):
  combined_speed_data = np.concatenate((combined_speed_data, st), axis=0)
  combined_power_data = np.concatenate((combined_power_data, pt), axis=0)
combined_power_data = combined_power_data.reshape(-1,1)
combined_speed_data = combined_speed_data.reshape(-1,1)
print(combined_speed_data.shape)
model = ps.SINDy(feature_library =ps.PolynomialLibrary(degree=2), differentiation_method=SINDyDerivative(kind="trend_filtered"), optimizer=ps.SR3(threshold=0.0001, thresholder="l1", normalize_columns=True), feature_names = ['P', 'S'], discrete_time=True)


model.fit(combined_power_data, u=combined_speed_data)

model.print()

def create_sindy_model_with_control(model):
    def sindy_model_with_control(interval: List[Timestamp], init_val: float, control_values: np.array):
        values = [init_val]
        degree = model.feature_library.degree
        feature_names = model.feature_names
        coefficients = model.coefficients()
        interval_secs = [t.to_secs() for t in interval]
        feature_combinations = []
        print(coefficients)
        feature_dict = {feature: (values[-1] if feature == feature_names[0] else control_values[j-1]) for j, feature in enumerate(feature_names)}
        for d in range(1, degree + 1):
            for combo in combinations_with_replacement(feature_names, d):
                feature_combinations.append(combo)
        for i, (t, u) in enumerate(zip(interval_secs[1:], control_values[1:])):
            new_value = coefficients[0, 0]
            for j, combo in enumerate(feature_combinations, start=1):
                term_value = 1
                for feature in combo:
                    term_value *= feature_dict[feature]
                new_value += coefficients[0, j] * term_value
            values.append(new_value)
            feature_dict = {feature: (values[-1] if feature == feature_names[0] else control_values[i+1]) for j, feature in enumerate(feature_names)}
        return np.array(values)
    return sindy_model_with_control

sindy_flow = FlowCondition(0, create_sindy_model_with_control(model))

powersindy = RealValuedVar([sindy_flow], [], model2distr, label='P')
speedsindy = RealValuedVar([sindy_flow], [], model2distr, label='w')
energy_made_cs = SystemUnderLearning([powersindy, speedsindy], events, parse_data, label_event, get_power_param, is_chg_pt, args=args)
'''END'''
test = False
if test:
    TEST_PATH = '/home/simo/WebFarm/lsha/resources/traces/MADE/'
    traces_files = os.listdir(TEST_PATH)

    for file in traces_files:
        # testing data to signals conversion
        new_signals: List[SampledSignal] = parse_data(TEST_PATH + file)
        # testing chg pts identification
        chg_pts = energy_made_cs.find_chg_pts([sig for sig in new_signals if sig.label in DRIVER_SIG])
        # testing event labeling
        id_events = [label_event(events, new_signals, pt.t) for pt in chg_pts[:10]]
        # testing signal to trace conversion
        energy_made_cs.process_data(TEST_PATH + file)
        trace = energy_made_cs.timed_traces[-1]
        print(Trace(tt=trace))
        power_pts = new_signals[0].points
        speed_pts = new_signals[1].points
        pressure_pts = new_signals[2].points
        double_plot([pt.timestamp for pt in power_pts], [pt.value for pt in power_pts],
                    [pt.timestamp for pt in speed_pts], [pt.value for pt in speed_pts],
                    trace, title=file, filtered=True,
                    timestamps3=[pt.timestamp for pt in pressure_pts],
                    v3=[pt.value for pt in pressure_pts])
    
    # test segment identification
    test_trace = Trace(energy_made_cs.traces[0][0:8])
    print(test_trace)
    pdf = PdfPages('test_trace_segments_plots.pdf')
    segments, segments_control = energy_made_cs.get_segments(test_trace, control=True)
    
    for i, (segment, control) in enumerate(zip_longest(segments, segments_control, fillvalue=None)):
        plt.figure()

        times = [pt.timestamp.to_secs() for pt in segment]
        values = [pt.value for pt in segment]

        plt.plot(times, values, label='Main Signal', color='blue')

        #if control is not None:
         #   control_times = [pt.timestamp.to_secs() for pt in control]
          #  control_values = [pt.value for pt in control]
           # plt.plot(control_times, control_values, label='Control Signal', color='red')
        plt.title(f'Segment {i + 1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.legend()

        pdf.savefig()
        plt.close()
    
    pdf.close()
    segments = energy_made_cs.get_segments(test_trace)

    # test model identification
    TEACHER = Teacher(energy_made_cs)
    identified_model: FlowCondition = TEACHER.mi_query(test_trace)

    print(identified_model)
    pdf = PdfPages('test_trace_identified.pdf')
    for i, (segment, control) in enumerate(zip_longest(segments, segments_control, fillvalue=None)):
        plt.figure()

        ts = [pt.timestamp for pt in segment]
        tsecond = [pt.timestamp.to_secs() for pt in segment]
        control_values = [s.value/1000 for s in segments_control[i]]
        values = identified_model.f(ts, segment[i].value, control_values)
        
        plt.plot(tsecond, values, label='Main Signal', color='blue')

        #if control is not None:
         #   control_times = [pt.timestamp.to_secs() for pt in control]
          #  control_values = [pt.value for pt in control]
           # plt.plot(control_times, control_values, label='Control Signal', color='red')
        plt.title(f'Segment {i + 1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.legend()

        pdf.savefig()
        plt.close()
    
    pdf.close()
    print(model2distr)

    # test distr identification
    for i, trace in enumerate(TEACHER.timed_traces):
        for j, event in enumerate(trace.e):
            test_trace = Trace(energy_made_cs.traces[i][:j + 1])
            identified_distr = TEACHER.ht_query(test_trace, identified_model, save=True)

            segments = energy_made_cs.get_segments(test_trace)
            avg_metrics = sum([TEACHER.sul.get_ht_params(segment, identified_model)
                               for segment in segments]) / len(segments)

            try:
                print('{}:\t{:.3f}->{}'.format(test_trace.events[-1].symbol, avg_metrics, identified_distr.params))
            except IndexError:
                print('{}:\t{:.3f}->{}'.format(test_trace, avg_metrics, identified_distr.params))
