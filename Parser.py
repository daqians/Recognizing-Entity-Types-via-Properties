# Import libraries
from rdflib import Graph, Namespace
from owlready2 import get_ontology
from rdflib.util import guess_format
from rdflib.plugins.parsers.notation3 import N3Parser
import pandas as pd
import os
import time
import re

class ExcelFile:
    def __init__(self, writer, num=0):
        self.writer = writer
        self.index = 1
        self.num = num + 1


def newExcel(excelNum, fileName, sheetName):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(fileName, engine='xlsxwriter',
                            options={'strings_to_urls': False, 'constant_memory': True, 'nan_inf_to_errors': True})
    excelFile_ = ExcelFile(writer, excelNum)
    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    # Add WorkSheet with relative titles and relative bold header
    worksheet = workbook.add_worksheet(sheetName)
    worksheet.write_row(0, 0, (
        "Date", "Subject", "Predicate", "Object", "SubjectTerm", "PredicateTerm", "ObjectTerm", "Domain",
        "Domain Version",
        "Domain Date", "URI", "Title", "Languages"), workbook.add_format({"bold": True}))
    worksheet.set_column(0, 8, 30)
    # Return the new excelFile_, workbook, worksheet
    return excelFile_, workbook, worksheet

def newExcel_easy(excelNum, fileName, sheetName):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(fileName, engine='xlsxwriter',
                            options={'strings_to_urls': False, 'constant_memory': True, 'nan_inf_to_errors': True})
    excelFile_ = ExcelFile(writer, excelNum)
    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    # Add WorkSheet with relative titles and relative bold header
    worksheet = workbook.add_worksheet(sheetName)
    worksheet.write_row(0, 0, (
        "SubjectTerm", "PredicateTerm", "ObjectTerm"), workbook.add_format({"bold": True}))
    worksheet.set_column(0, 8, 30)
    # Return the new excelFile_, workbook, worksheet
    return excelFile_, workbook, worksheet

# Read the the multiEtype object of a triple by owlready2
def get_from_owlReady(property, predicate,onto):
    def formalize_multiEtypes(items):
        try:
            s = ""
            for i in items:
                for j in str(i).split("|"):
                    EtypeName = j.strip().split(".")[-1]
                    s = s + EtypeName + '|'
            return s[:-1]
        except:
            return "BNode"

    try:
        kk = onto.search(iri=str(property))[0]
        Flagsubclass = str(predicate).lower()
        if predicate == "domain":
            Etypes = kk.domain
            ss = formalize_multiEtypes(Etypes)
        elif predicate == "range":
            Etypes = kk.range
            ss = formalize_multiEtypes(Etypes)
        elif "subclass" in Flagsubclass:
            Etypes = onto.search(subclass_of=kk)
            ss = formalize_multiEtypes(Etypes)
        else: return "BNode"
        return ss

    except:
        # print(property, predicate)
        return "BNode"


def get_label(iri,onto):
    try:
        kk = onto.search(iri="*" + iri)
        label = kk[0].label[0]
        return label
    except:
        return str(iri)

def get_label_multiEtypes(objectTerm,onto):

    try:
        s = ""
        for iri in objectTerm.split("|"):
            kk = onto.search(iri="*" + iri)
            label = kk[0].label[0]
            s = s + str(label) + "|"
        return s[:-1]
    except:
        return objectTerm


# Read all triples by rdflib
def parse(FileName,path,store):
    # Try to create the graph to analyze the vocabulary
    ONTaddress = path + FileName
    date = time.strftime("%Y-%m-%d", time.gmtime())
    print(ONTaddress)
    strPredicates = ["domain", "domainIncludes", "range"]
    try:
        g = Graph()
        format_ = FileName.split(".")[-1].split("?")[0]
        g.parse(ONTaddress, format=guess_format(format_))
        print("load " + ONTaddress + "by rdflib")

    except Exception as e:
        # In case of an error during the graph's initiation, print the error and return an empty list
        print(str(e) + "\n")
        return

    # Elaborate the fileName of the vocabulary
    fileName = FileName.split(".")[0]
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    singleExcel, singleWorkbook, singleSheet = newExcel(0, str(os.path.join(store, fileName + ".xlsx")),
                                                        "Single Parsed Triples")

    # singleExcel_FCA, singleWorkbook_FCA, singleSheet_FCA = newExcel_easy(0, str(os.path.join(store, fileName + "_Easy.xlsx")),
    #                                                          "Single Parsed Triples")
    # For each statement present in the graph obtained store the triples
    onto = get_ontology(ONTaddress).load()
    print("load "+ ONTaddress + "by owlready2")
    for subject, predicate, object_ in g:
        CNodes = ['ERO', 'OBI', 'BFO', 'IAO', 'RO', 'SWO', 'ERO', 'ARG', 'FLOPO','PATO','PO','ENVO']

        # Compute the filtered statement of the Triples
        subjectTerm = subject.replace("/", "#").split("#")[-1]
        if not len(subjectTerm) and len(subject.replace("/", "#").split("#")) > 1:
            subjectTerm = subject.replace("/", "#").split("#")[-2]

        predicateTerm = predicate.replace("/", "#").split("#")[-1]
        if not len(predicateTerm) and len(predicate.replace("/", "#").split("#")) > 1:
            predicateTerm = predicate.replace("/", "#").split("#")[-2]

        if type(object_).__name__  == "BNode":
            objectTerm = get_from_owlReady(subject, predicateTerm,onto)
            # print(subjectTerm,predicateTerm,objectTerm)
        else:
            objectTerm = object_.replace("/", "#").split("#")[-1]
            if not len(objectTerm) and len(object_.replace("/", "#").split("#")) > 1:
                objectTerm = object_.replace("/", "#").split("#")[-2]

        for c in CNodes:
            if c in subjectTerm:
                subjectTerm = get_label(subject, onto)
            if c in predicateTerm:
                predicateTerm = get_label(predicate, onto)
            if c in objectTerm and "|" in objectTerm:
                objectTerm = get_label_multiEtypes(objectTerm, onto)
            elif c in objectTerm:
                objectTerm = get_label(object_, onto)

        # Check if the triple has to be saved
        if True:
            # Save the statement to the ExcelSheet Triples
            singleSheet.write_row(singleExcel.index, 0, (
                date, subject, predicate, object_, subjectTerm, predicateTerm, objectTerm, FileName.split(".")[0],
                "v1.0", date, "URI", "Title", "en"))

            # Update the index of both the ExcelFiles
            singleExcel.index += 1
            if (singleExcel.index) % 1000 == 0:
                print(singleExcel.index)
                print(subject, predicate, object_)
                print(subjectTerm, predicateTerm, objectTerm)

            # If the rows reach the excel limit then create a new ExcelFile
            if singleExcel.index == 1048575:
                # Close the ExcelFile
                singleWorkbook.close()
                singleExcel.writer.save()
                # Create a new ExcelFile
                singleExcel, singleWorkbook, singleSheet = newExcel(singleExcel.num, str(
                    os.path.join(store, fileName + str(singleExcel.num) + ".xlsx")), "Single Parsed Triples")

        # if (str(predicateTerm) in strPredicates):
        #     # Easy Version for calculating FCAs
        #     if "|" in str(objectTerm):
        #         for obs in str(objectTerm).split("|"):
        #             singleSheet_FCA.write_row(singleExcel_FCA.index, 0, (
        #                 subjectTerm, predicateTerm, obs))
        #             singleExcel_FCA.index += 1
        #     else:
        #         singleSheet_FCA.write_row(singleExcel_FCA.index, 0, (
        #             subjectTerm, predicateTerm, objectTerm))
        #         singleExcel_FCA.index += 1


    # Close the Excel file of the single vocabulary
    # singleExcel_FCA.writer.book.close()
    # singleExcel_FCA.writer.save()
    singleExcel.writer.book.close()
    singleExcel.writer.save()

    g.close()
    # Return the List to be added to the DataFrame and the relative index
    return


def readFiles(tpath):
    txtLists = os.listdir(tpath)

    return txtLists


# path = readFiles("K-Files/inputs/")
DATASET = 'Conference'
path = "D:\Docs\programs\TrentoLab\OM\Data\General/generalOntology/"
Filenames = readFiles(path)
print(Filenames)
for Filename in Filenames:
    if ".rdf" not in Filename and ".owl" not in Filename:
        continue
    parse(Filename,path,store = 'triples/%s' %DATASET)
