%import common.SIGNED_NUMBER
%import common.NUMBER
%import common.NEWLINE
%import common.WS
%import common.UCASE_LETTER
%import common.LCASE_LETTER
%import common.WS_INLINE
%import common.CNAME
%import common.STRING_INNER
%import common.DIGIT
%ignore WS
start: lines+
?lines: age | sex | holter_rec | holter_rec_symp | comments | diagnosis | treatment | history
integer: NUMBER
age: "Age:"  (integer | text_body+)
?sex: "Sex:" sex_options
!sex_options: "F" | "M"
comments: "Comments:" (text_body+ | lead*)

//  leads
lead: "Lead" lead_number ":" text_body+
lead_number: integer

// holter
holter_rec_symp: "Symptoms during Holter recording:" text_body+
?text_body: /[\/a-zA-Z0-9-()%,;:><"'&@=. ]+/

diagnosis: "Diagnoses:" text_body+
treatment: "Treatment:" key_value_pairs+
negative: "No" | "Negative" |  "None"
positive: "Yes"
date: integer "/" integer "/" integer
key_value_pairs: text_body ":" (negative | positive | date | text_body+ | key_value_pairs+)
history:  "History:"  key_value_pairs+
holter_rec:  "Holter Recording:" key_value_pairs+