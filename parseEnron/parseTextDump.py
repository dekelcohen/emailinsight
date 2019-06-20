#!/usr/bin/env python
import sys
import io
import os
import os.path
import itertools
import json

import email

import dateutil.parser
import datetime
import pytz

import logging
import argparse


def to_camel_case(v):
    """
    Convert keys with hyphens to camelcase so we can use Dataset schema inference in Spark 2.0.
    See README.md for more details.
    """
    parts = v.split('-')
    return parts[0] + "".join(x.title() for x in parts[1:])


def parse_cats(f):
    """
    Format of each line in .cats file:
    n1,n2,n3

    n1 = top-level category
    n2 = second-level category
    n3 = frequency with which this category was assigned to this message
    """

    categories = []
    for line in f:
        cat_arr = line.rstrip('\r\n').split(',')
        categories.append({
            'n1': cat_arr[0],
            'n2': cat_arr[1],
            # Omitting frequency due to Spark Dataset nested array bug
            # https://issues.apache.org/jira/browse/SPARK-14034
            # 'frequency': cat_arr[2]
        })
    return categories


def parse_email(email_id, f):
    """
    :param email_id:
    :param f:
    :return:
    """
    raw = f.read().decode("utf-8")
    doc = {'id': email_id}

    # Parse the email using the Python email module and save fields
    e = email.message_from_string(raw)

    if e.is_multipart():
        logging.warn("Email is multipart, skipping.")
        return None

    # Add the email fields with lowercase keys to the document
    doc = dict([(k.lower(), v) for k, v in doc.items()] + [(k.lower(), v) for k, v in e.items()])

    # Standarize the date field to UTC
    doc['dateRaw'] = doc['date']
    doc['date'] = dateutil.parser.parse(doc['date'])
    doc['date'] = doc['date'].astimezone(pytz.utc).isoformat()

    # Clean up and split recipients fields
    for field in ['to', 'cc']:
        if field not in doc or doc[field] is None:
            doc[field] = []
        else:
            mails = doc.get(field, "").split(',')
            names = doc.get('x-' + field, "").split(',')
            n = len(mails)
            if (n > 0):
                doc[field] = [
                   {'id': mails[i], 'mail' : mails[i], 'displayName' : names[i], 'systemName': 'mail', 'personType': {"class": "Person", "subclass": "OrganizationUser"}}
                   for i in range(len(mails))
                ]

    fromAddr = doc.get('from', "")
    fromName = doc.get('x-from', "")
    doc['from'] = {'id': fromAddr, 'mail' : fromAddr, 'displayName' : fromName, 'systemName': 'mail', 'personType': {"class": "Person", "subclass": "OrganizationUser"}}

    # Add the payload
    doc['body'] = e.get_payload()

    # Remove newline from the subject and trim whitespace
    doc['subject'] = doc['subject'].replace('\r', '').replace('\n', '').strip()
    doc['body'] = doc['body'].replace('=20', '').replace('=09', '').strip()
    # camelcase keys for compatibility with Spark Dataset deserialization magic
    doc = {to_camel_case(k): v for k, v in doc.items()}
    deletedList = ['deleted', 'deleted items', 'deleted_items']
    sentitemsList = ['sent mail', 'sent items', 'sent']
    if (any(x in doc['xFolder'].lower() for x in sentitemsList)):
        doc['folderName'] = "Sent Items"
        doc['sourceId'] = "AQMkADMyZjllOTg2LWE0NjQtNDgzOS04MWJkLTg5Yjg5ZDQxOTBlMAAuAAADJe8a8wFYrESjZjxtDLv4JwEA3mjSq4rmW0GK7iFybG4bjQAAAWrSqgAAAA=="
    elif (any(x in doc['xFolder'].lower() for x in deletedList)):
        doc['folderName'] = "Deleted Items"
        doc['sourceId'] = "AQMkADMyZjllOTg2LWE0NjQtNDgzOS04MWJkLTg5Yjg5ZDQxOTBlMAAuAAADJeEU8wX29ESjZjxtDLv4JwEA3mjSq4rmW0GK7iFybG4bjQAAAWrSq27TZA=="
    else:
        doc['folderName'] = doc['xFolder'].rpartition('\\')[-1]
        doc['sourceId'] = "AAMkADQzNWU1MDk4LWVlNDUtNGRlZC1hY2E1LTY4MWJhYTdmNGE1YgAuAAAAAACXwS5KFVBzSLn7jlRp1ZgFAQCBvya2UjklRK8jVFIuQ0yUAAAAVxRmAAA="

    doc['updateId']  = doc['messageId']
    doc['conversationId']  = doc['messageId']
    doc['collageUserId']  = doc['xOrigin'].upper()
    doc['isFocused'] =  "true"
    doc['createdAt']  = doc['date']
    doc.pop('messageId', None)
    doc.pop('mimeVersion', None)
    doc.pop('contentType', None)
    doc.pop('xTo', None)
    doc.pop('xFrom', None)
    doc.pop('xCc', None)
    doc.pop('xBcc', None)
    doc.pop('xTo', None)
    doc.pop('dateRow', None)
    doc.pop('xOrigin', None)
    doc.pop('contentTransferEncoding', None)
    doc.pop('categories', None)
    doc.pop('xFolder', None)
    doc.pop('xFilename', None)
    doc.pop('dateRaw', None)
    return doc


def main():
    print ("Before parse")
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir')
    parser.add_argument('out_file')
    args = parser.parse_args()    
    print ("After parse")
	
    logging.basicConfig(level=logging.INFO)

    with io.open(args.out_file, 'w') as out:
        # Walk through all files within the directory
        for (root_path, _, filenames) in os.walk(args.in_dir):

            # Group files by the name without its file extension
            for (_, file_iter) in itertools.groupby(sorted(filenames), lambda x: os.path.splitext(x)[0]):
                files = list(file_iter)

                if files == ['categories.txt']:
                    # We'll skip categories.txt that ships with the tar.gz
                    continue

                try:
#                     Check if there's a file that ends with .cats
                    cats_filename = next((f for f in files if f.endswith('.cats')), None)
                    email_filename = next((f for f in files if f != cats_filename), None)
    
                    if email_filename is None:
                        logging.warn("Category file %s has no corresponding email file, continuing.", cats_filename)
                        continue
    
                    with io.open(os.path.join(root_path, email_filename), 'rb') as f:
                        doc = parse_email(email_filename, f)
    
                    if cats_filename is not None:
                        with io.open(os.path.join(root_path, cats_filename), 'rb') as f:
                            doc['categories'] = parse_cats(f)
                    else:
                        doc['categories'] = []
    
                    # Write output
                    if doc is not None:
                        json.dump(doc, out, ensure_ascii=False)
                        out.write('\n')
                    else:
                        logging.warn("Missing .txt or .cats file for: %s", list(files))
                except Exception as e:
                    logging.error("Failed to parse group: %s", files)
                    logging.exception(e)

        logging.info("Output written to %s", args.out_file)


if __name__ == '__main__':
    main()
