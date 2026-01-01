#just in case, contingency plan

from happytransformer import HappyTextClassification

happy_tc = HappyTextClassification("DISTILBERT", "distilbert-base-uncased-finetuned-sst-2-english")

#happy, should show up as POSITIVE
result_h = happy_tc.classify_text("This is the best day of my life!")
print(result_h)

#disgust, should show up as NEGATIVE
result_d = happy_tc.classify_text("That's the grossest-looking pizza I've ever had.")
print(result_d)