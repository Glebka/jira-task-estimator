import csv
import re
import bisect
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

csvFields = ['key','id','pid','summary','component','component','component','component',
                 'description','epic_link','epic_name','ext_ref',
                 'estimate','story_points']

categories = [0,1,2,3,5,8,13,21,34,55,89,144]

def __processText(value):
    if value:
        splitted = re.findall(r"[\w']+", value)
        splitted = map(str.lower, filter(lambda x: len(x) > 2, splitted))
        value = ' '.join(splitted)
    return value

def __processComponents(value):
    if value:
        if type(value) is list:
            value = map(str.lower, filter(len, value))
            value = ' '.join(value)
        else:
            value = value.lower()
    return value

def __processEstimate(value):
    if value:
        value = int(value) / 60 /60
        index = bisect.bisect(categories, value)
        if index < len(categories):
            value = str(categories[index])
        else:
            value = '?'
    return value

def __processStoryPoints(value):
    if value:
        index = bisect.bisect(categories, float(value))
        if index < len(categories):
            value = str(categories[index])
        else:
            value = '?'
    return value

normalizeMappings = {
    'summary': __processText,
    'component': __processComponents,
    'description': __processText,
    'estimate': __processEstimate,
    'story_points': __processStoryPoints
}

def makeDict(pairs):
    result = {}
    for pair in pairs:
        key = pair[0]
        value = pair[1]
        if key in result:
            if type(result[key]) is list:
                result[key].append(value)
            else:
                array = [result[key]]
                array.append(value)
                result[key] = array
        else:
            result[key] = value
    return result

def normalize(issue):
    for key, value in issue.iteritems():
        func = normalizeMappings.get(key)
        if func:
            issue[key] = func(value)
    doc = {
        'text': ' '.join([issue['component'],
                          issue['summary'],
                          issue['description']]),
        'category': issue['estimate'] if issue['estimate'] else issue['story_points']
    }
    return doc

def parseCsv():
    issues = []
    index = []
    with open('issues.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        isFirstRow = True
        for row in reader:
            if isFirstRow:
                isFirstRow = False
                continue
            keyValuePairs = zip(csvFields, row)
            issues.append(normalize(makeDict(keyValuePairs)))
            index.append(keyValuePairs[0][1])
    return DataFrame(issues, index)


def main():
    data = parseCsv()
    data = data.reindex(numpy.random.permutation(data.index))
    count_vectorizer = CountVectorizer()
    counts = count_vectorizer.fit_transform(data['text'].values)
    classifier = MultinomialNB()
    targets = data['category'].values
    classifier.fit(counts, targets)
    examples = []
    examples = map(__processText, examples)
    example_counts = count_vectorizer.transform(examples)
    predictions = classifier.predict(example_counts)
    print predictions
    print 'Done!'

if __name__ == "__main__":
    main()