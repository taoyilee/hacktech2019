%import common.SIGNED_NUMBER
%import common.NEWLINE
%import common.WS
%import common.LETTER
%import common.WORD
%import common.DIGIT
%ignore WS
start: record lines+
record: record_name number_of_signals sampling_frequency?
?lines: signal_spec | comment

CNAME: ("_"|LETTER) (LETTER|DIGIT|"_"|"-")*
record_name: CNAME
integer: SIGNED_NUMBER
number_of_signals: integer
sampling_frequency: integer counter_frequency? number_of_samples_per_signal?
counter_frequency: "/" integer base_counter_value?
base_counter_value: "(" integer ")"
number_of_samples_per_signal: integer base_time?
base_time: TIME_FORMAT base_date?
base_date: DATE_FORMAT
TIME_FORMAT: SIGNED_NUMBER ":" SIGNED_NUMBER ":" SIGNED_NUMBER
DATE_FORMAT:SIGNED_NUMBER "/" SIGNED_NUMBER "/" SIGNED_NUMBER

FILE_BASE_NAME: CNAME
dat_name: CNAME ".dat"
?file_name: dat_name
!signal_format: "8" | "16" | "24" | "32" | "61" | "80" | "160" | "212" | "310" | "311"
samples_per_frame: integer "x"
skew: integer ":"
byte_offset: integer "+"
adc_gain: integer baseline_adc? adc_units? adc_resol?
baseline_adc: "(" integer ")"
adc_units: "/" WORD
adc_resol: integer adc_zero?
adc_zero: integer adc_init_val?
adc_init_val: integer adc_checksum?
adc_checksum: integer adc_block_size?
adc_block_size: integer
signal_description: signal_description_options " "*
!signal_description_options: CNAME

signal_spec: file_name signal_format samples_per_frame? skew? byte_offset? adc_gain? signal_description?
comment: /#[^\n]*/