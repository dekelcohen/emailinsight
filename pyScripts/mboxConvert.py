import mailbox
import os
import re
import csv
import dateutil.parser
import pandas as pd


class parsedEmail():

    def __init__(self, updateId, label,subject,sender,fromDomain,timeRec,content,words=None, to=None,toDomain=None,cc=None,ccDomain=None):
        self.updateId = updateId
        self.label = label
        self.subject = subject
        self.sender = sender
        self.fromDomain = fromDomain
        self.to = to
        self.toDomain = toDomain
        self.cc = cc
        self.ccDomain = ccDomain
        self.day = timeRec[0]
        self.date = timeRec[1]
        self.month = timeRec[2]
        self.year = timeRec[3]
        self.hour = timeRec[4]
        self.content = content
        self.words = words        

    def __iter__(self):
        return iter([self.updateId, self.label, self.subject, self.sender, self.fromDomain, self.to,self.toDomain,
                     self.cc, self.ccDomain, self.day, self.date, self.month, self.year, self.hour, self.content, self.words])

evilSubstringsRegex = ['<html>.*</html>',\
                       '=20(.*\n)*=20',\
                       '\<.*\>',\
                        '\>.*\n',\
                        'Content\-.*[ \n]']
def cleanEmail(string):
    for evilRegex in evilSubstringsRegex:
        string = re.sub(evilRegex,'',string)
    return string

def addToCountDict(word,countDict):
    if word in countDict:
        countDict[word]+=1
    else:
        countDict[word]=1

# Parse CSV (Ex: Enron emails from mongo) that have different (but similar) set of attributes
# Format: Tabbed delimited .tsv with folderName	updateId	subject	body	from	fromDomain	to	cc	date). fold
# folderName- category / class 
# updateId - String - not used in classfication 
# subject - String - email subject  
# body - String - email body 
# from - String - displayName + emailAddress of sender (can experiment with only emailAddress or only displayName)        
# fromDomain - String email domain of sender (enron.com)
# to,cc - String - contactination of email to (see from field above)        
# date - ISO format date  ex: 1999-12-13T11:33:00+00:00   
def parseEmailsCSV(csvEmailsFilePath,printInfo=True):
    emails = []
    with open(csvEmailsFilePath) as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        count = 0;
        for row in reader:
            email = parseEmailCSV(row)
            emails.append(email)
            if printInfo:
                print('Parsed %d emails\n'%count)
    return emails

def parseEmailCSV(email):
    updateId = email['updateId']
    category = email['folderName']
    subject = email['subject']
    body = email['body']
    sender = email['from']
    fromDomain = email['fromDomain']    
    try:
        date = dateutil.parser.parse(email['date'])
    except:
        print('except: row[date]='%str(email['date']))
    if date is None:
        return
    dateParts = [date.day,date,date.month,date.year,date.hour]
    # dict of <word,count>
    splitted  = body.split(" ")
    messageWords = list(filter(None,splitted))
    wordCount = {}
    for word in messageWords:
        if word=='-----Original Message-----':
            break
        if len(word)==0 or '\r' in word or '=' in word \
           or '#' in word or '&' in word or word[0].isupper():
            continue
        word = re.sub('[|;\"\'\>\<\'\)\(,.?!\n]','',word)
        if len(word)>3:
            addToCountDict(word,wordCount)
    parsed_email = parsedEmail(updateId, category,subject,sender,fromDomain,dateParts,body,wordCount, to=email['to'], toDomain=email['toDomain'], cc=email['cc'], ccDomain=email['ccDomain'])
    return parsed_email

def addColumnsCSV(emails):
    days = []
    months = []
    years = []
    hours = []
    words = []
    for index, email in emails.iterrows():
        try:
            date = dateutil.parser.parse(email['date'])
        except:
            print('except: row[date]='%str(email['date']))
        if date is None:
            return
        dateParts = [date.day,date,date.month,date.year,date.hour]
        days.append(date.day)
        months.append(date.month)
        years.append(date.year)
        hours.append(date.hour)
        # dict of <word,count>
        splitted  = email['content'].split(" ")
        messageWords = list(filter(None,splitted))
        wordCount = {}
        for word in messageWords:
            if word=='-----Original Message-----':
                break
            if len(word)==0 or '\r' in word or '=' in word \
               or '#' in word or '&' in word or word[0].isupper():
                continue
            word = re.sub('[|;\"\'\>\<\'\)\(,.?!\n]','',word)
            if len(word)>3:
                addToCountDict(word,wordCount)
        words.append(wordCount)
    emails["day"] = days
    emails["month"] = months
    emails["year"] = years
    emails["hour"] = hours
    emails["words"] = words

def parseEmails(folder,printInfo=True):
    files = os.listdir(folder)
    emails = []
    for aFile in files:
        if os.path.isdir(aFile):
            emails += parseEmails(folder+'/'+aFile)
        elif aFile.endswith('.mbox'):
            box = mailbox.mbox(folder+'/'+aFile)
            count = 0
            if printInfo:
                print ('Parsing %s'%aFile)
            for message in box:
                if message['X-Gmail-Labels'] is None:
                    continue		      
                labels = message['X-Gmail-Labels'].split(',')
                if 'Chat' in labels:
                    continue
                if len(labels)>1 and 'Important' in labels:
        	            labels.remove('Important')
                if len(labels)>1 and 'Sent' in labels:
        	            labels.remove('Sent')
                if len(labels)>1 and 'Financial' in labels:
        	            labels.remove('Financial')
                if len(labels)>1 and 'Starred' in labels:
        	            labels.remove('Starred')
                category = labels[0]
                subject = message['subject']
                updateId = message['updateId']
                try:
                  sender = re.sub('[\-=|;\"\>\<\'\)\(,.?!\n\r\t]','',message['from'])
                except:
                    if '@' not in message['from']:
                        continue
                    senderDomain = sender[sender.index('@'):]
                    
                    date = message['Date']
                    if date is None:
                        continue
                        dateParts = date.split(" ")
                        dateParts[0] = dateParts[0][:-1]
                        dateParts[4] = dateParts[4][:2]
                        
                        payload = message.get_payload()
                        if message.is_multipart():
                            messageContent = payload[0].as_string()
                        else:
                            messageContent = payload
                        messageContent = cleanEmail(messageContent)
                        if len(messageContent)>10000:
                            continue
        
                        messageWords = messageContent.split(" ")
                        
                        wordCount = {}
                        for word in messageWords:
                            if word=='-----Original Message-----':
                                break
                            if len(word)==0 or '\r' in word or '=' in word \
                               or '#' in word or '&' in word or word[0].isupper():
                                continue
                            word = re.sub('[|;\"\'\>\<\'\)\(,.?!\n]','',word)
                            if len(word)>3:
                                addToCountDict(word,wordCount)
                        count+=1
                        email = parsedEmail(updateId, category,subject,sender,senderDomain,\
                                                dateParts,messageContent,wordCount)
                        emails.append(email)
                    if printInfo:
                        print ('Parsed %d emails\n'%count)
            return emails

def getEmailStats(emails):
    fromCount = {}
    domainCount = {}
    totalWordsCounts = {}
    labels = []
    for index, email in emails.iterrows():
        addToCountDict(email.sender,fromCount)
        addToCountDict(email.fromDomain,domainCount)
        if email.label not in labels:
            totalWordsCounts[email.label] = {}
            labels.append(email.label)

        words = email.words
        if type(words) == str:
            words = eval(words)
        for word in words:
            if word not in totalWordsCounts and word not in labels:
                totalWordsCounts[word]=0
            if word not in totalWordsCounts[email.label]:
                totalWordsCounts[email.label][word]=0
            if word not in labels:
                totalWordsCounts[word]+=words[word]
            totalWordsCounts[email.label][word]+=words[word]
    return (totalWordsCounts,fromCount,domainCount,labels)

def getTopEmailCounts(emails,percentThresh=0.25,numWords=50,numSenders=20,numDomains=5,perLabel=False):
    (totalWordsCount,fromCount,domainCount,labels) = getEmailStats(emails)
    topWords = set([])
    if perLabel:
       for label in labels:
            labelWords = totalWordsCount[label]
            topWordsDict = {}
            for word in labelWords:
                if word not in labels and labelWords[word]>totalWordsCount[word]*percentThresh:
                    topWordsDict[word] = labelWords[word]
            sortedWords = sorted(topWordsDict,key=topWordsDict.get,reverse=True)
            for i in range(min(numWords,len(sortedWords))):
                topWords.add(sortedWords[i])
    else:
        for label in labels:
            del totalWordsCount[label]
            sortedWords = sorted(totalWordsCount,key=totalWordsCount.get,reverse=True)
    topWords = sortedWords[:numWords*len(labels)]

    sortedSenders = sorted(fromCount,key=fromCount.get,reverse=True)
    sortedDomains = sorted(domainCount,key=domainCount.get,reverse=True)
    print ('%d words found.'%len(totalWordsCount))

    topSenders = sortedSenders[:numSenders]
    topDomains = sortedDomains if len(sortedDomains)<=numDomains else sortedDomains[:numDomains]
    return (topWords,topSenders,topDomains)

def mboxToBinaryCSV(folder,csvfile='data.csv',perLabel=True):
    outputFile = open(os.path.join(folder,csvfile),'w')

    emails = parseEmails(folder)
    (topWords,topSenders,topDomains)=getTopEmailCounts(emails,perLabel=perLabel)

    for sender in topSenders:
        outputFile.write('Sender %s,'%sender)
    for domain in topDomains:
        outputFile.write('From domain %s,'%domain)
    for word in topWords:
        outputFile.write('Has %s,'%word)
    outputFile.write('label\n')
    labelMap= {}
    for email in emails:
        for sender in topSenders:
            outputFile.write('1, ' if email.sender==sender else '0,')
        for domain in topDomains:
            outputFile.write('1, ' if email.fromDomain==domain else '0,')
        for word in topWords:
            outputFile.write('1, ' if word in email.words else '0,')
        if email.label not in labelMap:
            labelMap[email.label] = len(labelMap.keys())
        outputFile.write(str(labelMap[email.label])+'\n')
    outputFile.close()
    outputInfoFile = open(os.path.join(folder,csvfile+' info.txt'),'w')
    outputInfoFile.write('Num labels: %d'%len(labelMap.keys()))
    outputInfoFile.write('Label map: %s'%str(labelMap))
    outputInfoFile.close()

def mboxToCSV(folder, name='email.csv', limitSenders=True, limitDomains=True,perLabel=False):
    outputFile = open(folder+'/'+name,'w')

    emails = parseEmails(folder)
    (topWords,topSenders,topDomains)=getTopEmailCounts(emails,perLabel=perLabel)
    outputFile.write('Sender,Domain,')
    for word in topWords:
        outputFile.write('Has %s,'%word)
    outputFile.write('label\n')

    for email in emails:
        if not limitSenders or email.sender in topSenders:
            outputFile.write('%s,'%(email.sender))
        else:    
            outputFile.write('UncommonSender,')
        if not limitDomains or email.fromDomain in topDomains:
            outputFile.write('%s,'%(email.fromDomain))
        else:
            outputFile.write('UncommonDomain,')
        for word in topWords:
            outputFile.write('Yes,' if word in email.words else 'No,')
        outputFile.write(email.label+'\n')
    outFile.close()
