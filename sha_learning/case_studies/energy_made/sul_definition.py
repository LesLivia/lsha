from collections import defaultdict
import configparser
from itertools import combinations_with_replacement, zip_longest
import json
from matplotlib.backends.backend_pdf import PdfPages
import os
from typing import List, Tuple

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

TEST_PATH = config['TRACE GENERATION']['TEST_PATH']

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
support_cs = SystemUnderLearning([power, speed], events, parse_data, label_event, get_power_param, is_chg_pt, args=args)

# PySindy Flow Conditions
def extractTimestamps(points):
  return [str(point.timestamp).split(' ', 1)[1] for point in points]

def transform_times_to_seconds_cumulative(times):
    # Converte i tempi nel formato 'HH:MM:SS' in secondi totali
    times_seconds = [sum(int(x) * 60**i for i, x in enumerate(reversed(time.split(':')))) for time in times]
    # Calcola il tempo cumulativo trascorso dal primo elemento
    times_transformed = [time - times_seconds[0] for time in times_seconds]
    return np.array(times_transformed)

def generateData(data_paths):
    event_segments = []
    speed_segments = []
    power_segments = []
    event_symbol = []
    for data_path in data_paths:
        new_signals: List[SampledSignal] = parse_data(data_path)
        chg_pts = support_cs.find_chg_pts([sig for sig in new_signals if sig.label in DRIVER_SIG])
        power_pts = new_signals[0].points
        speed_pts = new_signals[1].points
        pressure_pts = new_signals[2].points
        id_events = [label_event(events, new_signals, pt.t) for pt in chg_pts[:10]]
        support_cs.process_data(data_path)
        trace = support_cs.timed_traces[-1]

        power_data = np.array([pt.value for pt in power_pts]).ravel()
        speed_data = np.array([pt.value/1000 for pt in speed_pts]).ravel() 
        t_power = transform_times_to_seconds_cumulative(np.array(extractTimestamps(power_pts)))
        t_speed = transform_times_to_seconds_cumulative(np.array(extractTimestamps(speed_pts)))

        for i in range(0, len(chg_pts) - 1):
            cp_start = chg_pts[i]
            cp_end = chg_pts[i + 1]
            event = label_event(events, new_signals, cp_start.t)

            event_start_timestamp = cp_start.t
            event_end_timestamp = cp_end.t
            speed_segment = np.array([pt.value/1000 for pt in speed_pts if event_start_timestamp <= pt.timestamp < event_end_timestamp]).ravel()
            power_segment = np.array([pt.value for pt in power_pts if event_start_timestamp <= pt.timestamp < event_end_timestamp]).ravel()

            speed_segment_timestamp = [pt for pt in speed_pts if event_start_timestamp <= pt.timestamp < event_end_timestamp]
            power_segment_timestamp = [pt for pt in power_pts if event_start_timestamp <= pt.timestamp < event_end_timestamp]
            event_segments.append((event.label, event.symbol, speed_segment_timestamp, power_segment_timestamp))
            speed_segments.append(speed_segment)
            power_segments.append(power_segment)
            event_symbol.append(event.symbol)
    return event_segments, event_symbol, power_segments, speed_segments

def create_sindy_model_with_control(model):
    def sindy_model_with_control(interval: List[Timestamp], init_val: float, control_values: List[float]):
        values = [init_val]
        control_values = [c/1000 for c in control_values]
        degree = model.feature_library.degree
        feature_names = model.feature_names
        coefficients = model.coefficients()
        interval_secs = [t.to_secs() for t in interval]
        feature_combinations = []
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
        return values
    return sindy_model_with_control


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
# Trace First Geometry (fisrt part program)
# lm_66i_0m_94i_0m_34i_0m_59i_0m_59i_0m_57i_0u (11_jan_1)
# lm_66i_0m_66i_0m_66i_0m_94i_0m_34i_0m_59i_0m_57i_0u (12_jan_1)
# lm_66i_0m_94i_0m_34i_0m_59i_0m_57i_0u (tutte le altre)
# Trace First Geometry (second part program)
# lm_41m_69i_0m_97i_0m_46i_0m_71i_0m_61i_0u (03_feb_2)
# lm_69i_0m_69i_0m_97i_0m_46i_0m_71i_0m_61i_0u (17_feb_1)
# lm_69i_0m_97i_0m_46i_0m_71i_0m_61i_0u (tutti gli altri csv)
base_path_first_geometry="/home/simo/WebFarm/lsha/resources/traces/MADE/FirstGeometry/"
data_paths_first_geometry = [
                base_path_first_geometry+"11_jan_1.csv", # fisrt part program 
                base_path_first_geometry+"11_jan_2.csv", # fisrt part program
                base_path_first_geometry+"12_jan_1.csv", # fisrt part program
                base_path_first_geometry+"12_jan_2.csv", # fisrt part program
                base_path_first_geometry+"12_jan_3.csv", # fisrt part program
                base_path_first_geometry+"12_jan_4.csv", # fisrt part program
                base_path_first_geometry+"12_jan_5.csv", # fisrt part program
                base_path_first_geometry+"13_feb_1.csv",
                base_path_first_geometry+"03_feb_2.csv",
                base_path_first_geometry+"03_mar_1.csv",
                base_path_first_geometry+"13_feb_2.csv",
                base_path_first_geometry+"15_feb_1.csv",
                base_path_first_geometry+"17_feb_1.csv",
                base_path_first_geometry+"17_feb_2.csv",
                base_path_first_geometry+"17_feb_3.csv"
                ]
# Trace Second Geometry: lm_69i_0m_46i_0m_69i_0m_61i_0m_71i_0u
base_path_second_geometry="/home/simo/WebFarm/lsha/resources/traces/MADE/SecondGeometry/"
data_paths_second_geometry = [
                base_path_second_geometry+"05_may_1.csv",
                base_path_second_geometry+"05_may_2.csv",
                base_path_second_geometry+"12_apr_1.csv",
                base_path_second_geometry+"12_apr_2.csv"
                ]


def generateFlowConditions(data_paths):

    event_segments, event_symbols, power_segments, speed_segments = generateData(data_paths)

    flow_conditions: List[FlowCondition] = []
    max_length = max(max(len(segment) for segment in speed_segments),
                 max(len(segment) for segment in power_segments))
    def pad_sequences(sequences, max_length):
        padded_sequences = []
        for sequence in sequences:
            num_zeros = max_length - len(sequence)
            padded_sequence = np.pad(sequence, (0, num_zeros), 'constant')
            padded_sequences.append(padded_sequence)
        return padded_sequences


    speed_segments_padded = pad_sequences(speed_segments, max_length)
    power_segments_padded = pad_sequences(power_segments, max_length)

    speed_data = np.array(speed_segments_padded)
    power_data = np.array(power_segments_padded)

    from collections import Counter
    counters = Counter(event_symbols)
    listCounter = list(counters.items())
    print(listCounter)
    noRepetitions = list(set(event_symbols))
    print(noRepetitions)
    
    from collections import defaultdict
    event_map = defaultdict(lambda: {'speed': [], 'power': []})
    model_map = defaultdict()

    
    for i, val in enumerate(event_symbols):
        event_map[val]['speed'].append(speed_data[i])
        event_map[val]['power'].append(power_data[i])
    counterFlowConditions = 0
    for symbol in event_map:
        speed_data = event_map[symbol]['speed']
        power_data = event_map[symbol]['power']
        combined_speed_train_data = []
        combined_power_train_data = []
        for st,pt in zip(speed_data, power_data):
            combined_speed_train_data = np.concatenate((combined_speed_train_data, st), axis=0)
            combined_power_train_data = np.concatenate((combined_power_train_data, pt), axis=0)
        combined_power_train_data = combined_power_train_data.reshape(-1,1)
        combined_speed_train_data = combined_speed_train_data.reshape(-1,1)
        model = ps.SINDy(feature_library =ps.PolynomialLibrary(degree=2),optimizer=ps.SR3(threshold=0.0001, thresholder="l1"), feature_names = ['P', 'S'], discrete_time=False)
        model.fit(combined_power_train_data,u=combined_speed_train_data)
        model_map[symbol] = model
        flow_conditions.append(FlowCondition(counterFlowConditions, create_sindy_model_with_control(model)))
        counterFlowConditions+=1
    counterFlowConditions = 0
    for symbol, model in model_map.items():
        print(f"Symbol {symbol}")
        print(f"Flow Condition {counterFlowConditions}")
        model.print()
        print("-" * 180)
        counterFlowConditions += 1
    count = 0
    with open("/home/simo/WebFarm/lsha/resources/learned_sha/made/flow_conditions.txt", "w") as file:
        for symbol, model in model_map.items():
            file.write(f"{count}\n")
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                model.print()
            out = f.getvalue()
        
            file.write(out)
            count += 1
        
    return flow_conditions

firstMethod = False

if firstMethod:
    models: List[FlowCondition]  = generateFlowConditions(data_paths_first_geometry+data_paths_second_geometry)
else:
# Secondo metodo
#train_cs = SystemUnderLearning([power, speed], events, parse_data, label_event, get_power_param, is_chg_pt, args=args)
#TEST_PATH = '/home/simo/WebFarm/lsha/resources/traces/MADE/All/'
    traces_files = data_paths_first_geometry + data_paths_second_geometry
#traces_files = os.listdir(TEST_PATH)

    event_map = defaultdict(lambda: {'speed': [], 'power': []})
    for i,file in enumerate(traces_files):
        new_signals: List[SampledSignal] = parse_data(file)
        chg_pts = support_cs.find_chg_pts([sig for sig in new_signals if sig.label in DRIVER_SIG])
        id_events = [label_event(events, new_signals, pt.t) for pt in chg_pts[:10]]
        support_cs.process_data(file)
        word = support_cs.traces[i]
        w_prev = []
        s = ""
        for w in word.events:
            s = s + w.symbol
            values = []
            values_c = []
            segments, segments_control = support_cs.get_segments(w_prev+[w], control=True)
            w_prev = w_prev+[w]
            if len(segments)>0:
                for segment,segment_c in zip(segments, segments_control):
                    #v = [x.value for x in segment] #sindy su questi
                    #vc = [x.value for x in segment_c]
                    #values.append(v)
                    #values_c.append(vc)
                    #event_map[w.symbol]['speed'].append(vc)
                    #event_map[w.symbol]['power'].append(v)
                    for x,xc in zip(segment, segment_c):
                        event_map[w.symbol]['speed'].append(xc.value)
                        event_map[w.symbol]['power'].append(x.value)

    flow_conditions: List[FlowCondition] = []
    print("\nItems processed")
    counterFlowConditions = 0
    model_map = defaultdict()
    for key, val in event_map.items():
        X = np.array(val["power"])
        y = np.array(val["speed"])
        model = ps.SINDy(feature_library =ps.PolynomialLibrary(degree=2),optimizer=ps.SR3(threshold=0.0001, thresholder="l1"), feature_names = ['P', 'S'], discrete_time=False)
        model.fit(X,u=y)
        model_map[key] = model
        flow_conditions.append(FlowCondition(counterFlowConditions, create_sindy_model_with_control(model)))
        counterFlowConditions+=1

    counterFlowConditions = 0
    for symbol, model in model_map.items():
        print(f"Symbol {symbol}")
        print(f"Flow Condition {counterFlowConditions}")
        model.print()
        print("-" * 180)
        counterFlowConditions += 1
    count = 0
    with open("/home/simo/WebFarm/lsha/resources/learned_sha/made/flow_conditions.txt", "w") as file:
        for symbol, model in model_map.items():
            file.write(f"{count}\n")
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                model.print()
            out = f.getvalue()
        
            file.write(out)
            count += 1
    
    models = flow_conditions
'''
Items processed
l
m_66
i_0
m_94
m_34
m_59
m_57
u
m_69
m_97
m_46
m_71
m_61
m_41
'''




model_to_distr = {}
for m in models:
    model_to_distr[m.f_id] = []

powersindy = RealValuedVar(models, [], model_to_distr, label='P')
speedsindy = RealValuedVar(models, [], model_to_distr, label='w')

energy_made_cs = SystemUnderLearning([powersindy, speedsindy], events, parse_data, label_event, get_power_param, is_chg_pt, args=args)
#energy_made_cs = SystemUnderLearning([power, speed], events, parse_data, label_event, get_power_param, is_chg_pt, args=args)

def get_timed_trace(input_file_name: str):
    energy_made_cs.process_data(TEST_PATH.format(input_file_name))
    tt = energy_made_cs.timed_traces[-1]
    tt_tup: List[Tuple[str, str]] = []

    # if len(tt) > 0 and tt.t[0].min > 0:
    #    # FIXME: non mi convince.
    #    tt.t = [Timestamp(tt.t[0].year, tt.t[0].month, tt.t[0].day, tt.t[0].hour, 0, 0)] + tt.t
    #    tt.e = [Event('', '', 'i_0')] + tt.e

    for i, event in enumerate(tt.e):
        if event.symbol == 'i_0':
            e_sym = 'STOP'
        elif event.symbol == 'l':
            e_sym = 'LOAD'
        elif event.symbol == 'u':
            e_sym = 'UNLOAD'
        else:
            e_sym = event.symbol.split('_')[1]
        if i == 0:
            diff_t = 0.0
        else:
            diff_t = ((tt.t[i].to_secs() - tt.t[0].to_secs()) - (tt.t[i - 1].to_secs() - tt.t[0].to_secs())) / 60

        tt_tup.append((str(diff_t), e_sym))

    return tt_tup


file_names = [
    "_03_mar_1",
    "_05_may_1",
    "_05_may_2",
    "_11_jan_2",
    "_12_apr_1",
    "_12_apr_2",
    "_12_jan_2",
    "_12_jan_3",
    "_12_jan_4",
    "_12_jan_5",
    "_13_feb_1",
    "_13_feb_2",
    "_15_feb_1",
    "_17_feb_2",
    "_17_feb_3"
]


def save_timed_trace(tt_tup, file_path):
    with open(file_path, 'w') as file:
        json.dump(tt_tup, file)

for file_name in file_names:
    tt = get_timed_trace(file_name)
    save_timed_trace(tt, f"{file_name}_timed_trace.json")


#END
test = False
if test:
    TEST_PATH = '/home/simo/WebFarm/lsha/resources/traces/MADE/All/'
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
        control_values = [s.value for s in segments_control[i]]
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