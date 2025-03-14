import csv
import gc
import re
import os
import subprocess
import requests
import xml.etree.ElementTree as ET
import json
import pandas as pd
import urllib

import sys
sys.setrecursionlimit(3000)
non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), '')

import nltk
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

import spacy
#from spacy import displacy
nlp = spacy.load('en_core_web_md')
nlp.add_pipe('sentencizer', before='parser') # for spaCy 3.x.x

'''from datetime import *
from dateutil.easter import *
from dateutil.parser import *
from dateutil.relativedelta import *
from dateutil.rrule import *'''
#from file_io import *

# class Helper --------------------------------------------------
class Helper():

    @staticmethod
    def remove_subsection(text):

        try: text = text[0:text.index('References')]
        except: pass

        try: text = text[0:text.index('Notes')]
        except: pass

        try: text = text[0:text.index('External links')]
        except: pass
        
        return text


    @staticmethod
    def remove_heading(text):
        text = re.sub(r'==([\w|\s|=])*==', '', text)
        return text
        

    @staticmethod
    def unwiki(text):
        """
            remove wiki markup from the text.
        """
        text = re.sub(r'(?i)\{\{IPA(\-[^\|\{\}]+)*?\|([^\|\{\}]+)(\|[^\{\}]+)*?\}\}', lambda m: m.group(2), text)
        text = re.sub(r'(?i)\{\{Lang(\-[^\|\{\}]+)*?\|([^\|\{\}]+)(\|[^\{\}]+)*?\}\}', lambda m: m.group(2), text)
        text = re.sub(r'\{\{[^\{\}]+\}\}', '', text)
        text = re.sub(r'(?m)\{\{[^\{\}]+\}\}', '', text)
        text = re.sub(r'(?m)\{\|[^\{\}]*?\|\}', '', text)
        text = re.sub(r'(?i)\[\[Category:[^\[\]]*?\]\]', '', text)
        text = re.sub(r'(?i)\[\[Image:[^\[\]]*?\]\]', '', text)
        text = re.sub(r'(?i)\[\[File:[^\[\]]*?\]\]', '', text)
        #text = re.sub(r'\[\[[^\[\]]*?\|([^\[\]]*?)\]\]', lambda m: m.group(1), text)
        #text = re.sub(r'\[\[([^\[\]]+?)\]\]', lambda m: m.group(1), text)
        #text = re.sub(r'\[\[([^\[\]]+?)\]\]', '', text)
        text = re.sub(r'(?i)File:[^\[\]]*?', '', text)
        #text = re.sub(r'\[[^\[\]]*? ([^\[\]]*?)\]', lambda m: m.group(1), text)
        #text = re.sub(r"''+", '', text)
        #text = re.sub(r'(?m)^\*$', '', text)
  
        text = re.sub(r'\<.*\>', '', text)

        text = text.replace('\r', '')
        text = text.replace('\n', '').strip()
        
        return text

    @staticmethod
    def create_link(link, params):
        """
            create full link
                link: string - link without parameters
                params: dict - parameter dict
                return: string - full link created
        """

        param_string = ''
        for k, v in params.items():
            param_string += str(k) + '=' + str(v) + '&'

        link += '?' +  param_string
        link = link.strip('&')

        return link
    

    @staticmethod
    def remove_emojis(text):
        """
            remove emojis from Karim Omaya (stackoverflow.com)
                text: string - input
                return: string - text without emojis
        """
        emoj = re.compile('['
            u'\U0001F600-\U0001F64F'  # emoticons
            u'\U0001F300-\U0001F5FF'  # symbols & pictographs
            u'\U0001F680-\U0001F6FF'  # transport & map symbols
            u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
            u'\U00002500-\U00002BEF'  # chinese char
            u'\U00002702-\U000027B0'
            u'\U00002702-\U000027B0'
            u'\U000024C2-\U0001F251'
            u'\U0001f926-\U0001f937'
            u'\U00010000-\U0010ffff'
            u'\u2640-\u2642' 
            u'\u2600-\u2B55'
            u'\u200d'
            u'\u23cf'
            u'\u23e9'
            u'\u231a'
            u'\ufe0f'  # dingbats
            u'\u3030'
                          ']+', re.UNICODE)
        return re.sub(emoj, '', text)

    @staticmethod
    def cls():
        """
            clear console screen
        """
        os.system('cls' if os.name == 'nt' else 'clear')
# -----------------------------------------------------------

# class WikiRequest --------------------------------------------
class WikiRequest():
    #def __init__(self):
    #    pass

    @staticmethod
    def get_data_by_link(link, params = {}, data_format = 'xml', timeout = 45):
        """
            get data from a web link
                link: string - a web link
                return: string - web data
        """
        
        if (data_format not in ['xml', 'json']): data_format  = 'xml' # set default

        response = requests.get(link, params, timeout = timeout)
        if (data_format == 'xml'): response = ET.fromstring(response.text)
        elif (data_format == 'json'): response = json.loads(response.text)
        
        return response
# -----------------------------------------------------------

# class Item ------------------------------------------------
class Item():
    def __init__(self, wikidata_id = '', item_type = '', label = '', description = '', sitelink = '', aliases = [], \
                 instances = [], subclasses = [], parts = [], claims = []):

        self.wikidata_id = wikidata_id
        self.type = item_type
        self.label = label
        self.description = description
        self.sitelink = sitelink
        
        self.aliases = aliases
        self.instances = instances
        self.subclasses = subclasses
        self.parts = parts
        self.claims = claims
        
# -----------------------------------------------------------

# class Wikidata --------------------------------------------
class Wikidata():

    def get_item_by_id(self, wikidata_id, language = 'en', return_type = 'dict', pass_wiki_page = False):
        """
            get a Wikidata item by id (e.g. Q123)
                wikidata_id: string - Wikidata identifier
                language: string - language needed to return
                return_type: string - type of return data
                return: dict or class - return data   
        """

        item_dict = {}
        try:

            wikidata_id = wikidata_id[0].upper() + wikidata_id[1:]
            
            item_type = ''
            if (wikidata_id[0] == 'Q'): item_type = 'item'
            elif(wikidata_id[0] == 'P'): item_type = 'property'
            else: item_type = 'unknown'
            
            wikidata_root = self.get_wikidata_root(wikidata_id)

            # check redirects
            from_id, to_id = self.get_redirect_id(wikidata_root)
            if (from_id == wikidata_id and from_id != to_id): wikidata_id = to_id
            
            description = self.get_description(wikidata_root, language)
    
            if (pass_wiki_page == True):
                if ('disambiguation page' in description.lower() or 'wiki' in description.lower()): return {}

            label = self.get_label(wikidata_root, language)
            sitelink = self.get_sitelink(wikidata_root, language)
            
            sitelinks = self.get_sitelinks(wikidata_root, language)
            
            claims = self.get_claims(wikidata_root, wikidata_id)
            instances = self.get_instance_of(claims, language)
            subclasses = self.get_subclass_of(claims, language)
            parts = self.get_part_of(claims, language)
            aliases = self.get_alias(wikidata_root, language)
            
            item_dict['wikidata_id'] = wikidata_id
            item_dict['item_type'] = item_type
            item_dict['label'] = label
            item_dict['description'] = description
            item_dict['sitelink'] = sitelink
            item_dict['sitelinks'] = sitelinks
            
            item_dict['instances'] = instances
            item_dict['subclasses'] = subclasses
            item_dict['aliases'] = aliases
            item_dict['parts'] = parts
            item_dict['claims'] = claims 
        
        except Exception as e:
            #print('Error --- get_item_by_id: ', e)
            pass

        if (return_type == 'class'):
            return Item(item_dict['wikidata_id'], item_dict['item_type'], item_dict['label'], item_dict['description'], \
                        item_dict['sitelink'], item_dict['aliases'], item_dict['instances'], item_dict['subclasses'], \
                        item_dict['parts'], item_dict['claims'])
    
        return item_dict

    def get_item_by_title(self, title, title_language = 'en', language = 'en', return_type = 'dict'):
        """
            get a Wikidata item by title (e.g. Science)
                title: string - page title
                title_language: string - language project (e.g. en = English Wikipedia, fr = Frech Wikipedia, etc.)
                language: string - language needed to return
                return_type: string - type of return data
                return: dict or class - return data   
        """

        item_dict = {}
        try:
            title = title[0].upper() + title[1:]
            
            page = Wikipedia()
            root = page.get_data_by_title(title, language = title_language)
    
            wikidata_id = self.get_id(root)
            if (wikidata_id == '' or wikidata_id == None): return {}

            item_type = ''
            if (wikidata_id[0] == 'Q'): item_type = 'item'
            elif(wikidata_id[0] == 'P'): item_type = 'property'
            else: item_type = 'unknown'

            wikidata_root = self.get_wikidata_root(wikidata_id)
       
            description = self.get_description(wikidata_root, language)
            if ('disambiguation page' in description): return {}

            label = self.get_label(wikidata_root, language)
            sitelink = self.get_sitelink(wikidata_root, language)
            
            aliases = self.get_alias(wikidata_root, language)
            claims = self.get_claims(wikidata_root, wikidata_id)
            instances = self.get_instance_of(claims, language)
            subclasses = self.get_subclass_of(claims, language)
            parts = self.get_part_of(claims, language)    

            item_dict['wikidata_id'] = wikidata_id
            item_dict['item_type'] = item_type
            item_dict['label'] = label
            item_dict['description'] = description
            item_dict['sitelink'] = sitelink
            
            item_dict['instances'] = instances
            item_dict['subclasses'] = subclasses
            item_dict['aliases'] = aliases
            item_dict['parts'] = parts
            item_dict['claims'] = claims
            
        except Exception as e:
            print('Error --- get_item_by_title: ', e)
            pass

        if (return_type == 'class'):
            return Item(item_dict['wikidata_id'], item_dict['item_type'], item_dict['label'], item_dict['description'], \
                        item_dict['sitelink'], item_dict['aliases'], item_dict['instances'], item_dict['subclasses'], \
                        item_dict['parts'], item_dict['claims'])
    
        return item_dict

    def get_wikidata_root(self, wikidata_id):
        """
            get XML data by if from Wikidata
                wikidata_id: string - Wikidata identifier
                return: return: string - XML data
        """
    
        if (wikidata_id == None or wikidata_id == ''): return ''
    
        link = 'https://www.wikidata.org/w/api.php'
        params = {
            'action': 'wbgetentities',
            'format': 'xml',
            'ids': wikidata_id,
            }
    
        root = WikiRequest.get_data_by_link(link, params)
        if (root is None): return ''
    
        return root

    def get_property_datatype(self, root):
        """
            get the datatype from XML data
                root: string - XML data
                return: string - a property datatype
        """
    
        try:
            for x in root.find('./entities'):
                value = Helper.remove_emojis(x.attrib['datatype'])
                if (value != ''): return value
        except: pass
        return ''               

    def get_sitelink(self, root, language = 'en'):
        """
            get a sitelink of a Wikipedia project from the Wikidata's XML data
                root: string - XML data of Wikidata
                wiki: string - a shortcut for a Wikipedia project
                return: string - a sitelink
        """
        wiki = language + 'wiki'
        try:
            for x in root.find('./entities/entity/sitelinks'):
                if (x.attrib['site'] == wiki):
                    value = Helper.remove_emojis(x.attrib['title'])
                    if (value != ''): return value
        except: pass
        return ''

    
    def get_sitelinks(self, root, language = 'en'):
        """
            get a sitelink of a Wikipedia project from the Wikidata's XML data
                root: string - XML data of Wikidata
                wiki: string - a shortcut for a Wikipedia project
                return: string - a sitelink
        """
        
        sitelink_dict = {}
        try:
            for x in root.find('./entities/entity/sitelinks'):
                #if (x.attrib['site'] == wiki):
                value = Helper.remove_emojis(x.attrib['title'])
                if (value != ''):
                    sitelink_dict[x.attrib['site']] = value
                    #return value
        except: pass
        return sitelink_dict
        
    def get_id(self, root):
        """
            get an identifier of an Wikidata item
                root: string - XML data of Wikidata
                return: string - wikidata_id
        """
    
        for node in root.find('./query/pages/page'):
            if (node.tag == 'pageprops'): return node.attrib['wikibase_item']
        return ''

    def get_label_by_id(self, wikidata_id, data_format = 'xml'):
        """
            get label by a Wikidata indenfifier
                wikidata_id: string - Wikidata identifier
                return: string - item label
        """
    
        link = 'https://www.wikidata.org/w/api.php'
        params = {
            'action': 'wbgetentities',
            'format': data_format,
            'ids': wikidata_id
            }
    
        root = WikiRequest.get_data_by_link(link, params)           
        return self.get_label(root)


    def get_label(self, root, language = 'en'):
        """
            get label from XML data
                root: string - XML data
                language: string - a language syntax
                return: string - label
        """
    
        try:
            for x in root.find('./entities/entity/labels'):
                if (language == 'en'):
                    if (x.attrib['language'] == language + '-gb' or x.attrib['language'] == language):
                        value = Helper.remove_emojis(x.attrib['value'])
                        if (value != ''): return value
                else:
                    if (x.attrib['language'] == language):
                        value = Helper.remove_emojis(x.attrib['value'])
                        if (value != ''): return value
        except:
            pass
        return ''

    def get_description(self, root, language = 'en'):
        """
            get label from XML data
                root: string - XML data
                language: string - a language syntax
                return: string - description
        """

        try:
            for x in root.find('./entities/entity/descriptions'):
                if (language == 'en'):
                    if (x.attrib['language'] == language + '-gb' or x.attrib['language'] == language):
                        value = Helper.remove_emojis(x.attrib['value'])
                        if (value != ''): return value
                else:
                    if (x.attrib['language'] == language):
                        value = Helper.remove_emojis(x.attrib['value'])
                        if (value != ''): return value
        except:
            pass
        return ''  

    def get_redirect_id(self, root):
        """
            check root containing new wikidata_id or not
                root: string - XML data
                
                return: string - new wikidata_id
        """

        try:
            for x in root.find('./entities/entity'):
                if (x.tag == 'redirects'):
                    return x.attrib['from'], x.attrib['to']
        except Exception as e:
            pass
        return '', ''

    def get_value_by_property(self, claim_list, property_name, language = 'en'):
        """
            get values of a given property
                claim_list: list - Wikidata claims
                property_name: string - a Wikidata property name
                return: list - a list of properties
        """
    
        result_list = []
        try:
            for c in claim_list:
                if (c[1][1] == property_name):
                    k = c[1][3] # wdid
                    root = self.get_wikidata_root(k)
                    label = self.get_label(root, language)
                    result_list.append([k, label])
        except: pass

        return result_list

    def get_instance_of(self, claim_list, language = 'en'):
        """
            get instance of a Wikidata item from its claims
                claim_list: list - Wikidata claims
                return: list - a list of instances
        """
    
        result_list = []
        try:
            for c in claim_list:
                if (c[1][1] == 'P31'):
                    k = c[1][3]
                    root = self.get_wikidata_root(k)
                    instance_name = self.get_label(root, language)
                    result_list.append([k, instance_name])
        except: pass

        return result_list
    
    def get_subclass_of(self, claim_list, language = 'en'):
        """
            get instance of a Wikidata item from its claims
                claim_list: list - Wikidata claims
                return: list - a list of instances
        """
    
        result_list = []
        try:
            for c in claim_list:
                if (c[1][1] == 'P279'):
                    k = c[1][3]
                    root = self.get_wikidata_root(k)
                    instance_name = self.get_label(root, language)
                    result_list.append([k, instance_name])
        except: pass

        return result_list

    def get_part_of(self, claim_list, language = 'en'):
        """
            get parts of a Wikidata item from its claims
                claim_list: list - Wikidata claims
                return: list - a list of parts
        """
    
        result_list = []
        try:
            for c in claim_list:
                if (c[1][1] == 'P361'):
                    k = c[1][3]
                    root = self.get_wikidata_root(k)
                    instance_name = self.get_label(root, language)
                    result_list.append([k, instance_name])
        except: pass

        return result_list
    

    def get_nationality(self, claim_list, language = 'en'):
        """
            get nationality of a Wikidata item from its claims
                claim_list: list - Wikidata claims
                return: list - a list of nationalities
        """
    
        result_list = []
        try:
            for c in claim_list:
                if (c[1][1] == 'P27'):
                    k = c[1][3]
                    root = self.get_wikidata_root(k)
                    country_name = self.get_label(root, language)
                    result_list.append([k, country_name])
        except: pass

        return result_list
	
    def get_alias(self, root, language = 'en'):
        """
            get aliases of a Wikidata item from its claims
                root: string - xml data
                return: list - a list of aliases
        """
    
        aliases = []
        try:
            for x in root.find('./entities/entity/aliases'):
            
                if (language == 'en'):
                    if (x.attrib['id'] == language + '-gb' or x.attrib['id'] == language):
                        for t in x:
                            value = Helper.remove_emojis(t.attrib['value'])
                            if (value != ''): aliases.append(value)
                else:
                    if (x.attrib['id'] == language):
                        for t in x:
                            value = Helper.remove_emojis(t.attrib['value'])
                            if (value != ''): aliases.append(value)
        except: pass
    
        return aliases
        
    def get_claims(self, root, wikidata_id):
        """
            get all claims (or sometimes called statements) of a Wikidata item
                root: XML data
                wikidata_id: Wikidata identifier
                return: list - a list of claims
        """

        claim_list = [] # statement list
        if (root == '' or root is None): return claim_list
        if (wikidata_id == '' or wikidata_id is None): return claim_list

        s = wikidata_id # s: subject (item identifier, wikidata_id)
        p = ob = pt = pv = q = qt = qv = ''
        # p: predicate (property), ob: object (property value identifier)
        # pt: object type (property value type), pv: object value (property value)
        # q: qualifier, qt: qualifier type, qv: qualifier value

        # loop each predicate (property)
        for predicate in root.find('./entities/entity/claims'):        
            #print('************************')
            #print('Property: ', predicate.attrib['id'])
            p = Helper.remove_emojis(predicate.attrib['id']) # predicate (property)
            for claim in predicate.iter('claim'):
                pt = Helper.remove_emojis(claim[0].attrib['datatype']) # property type
                #print('+', pt)
                for obj in claim.find('mainsnak'):
                    try:
                        try:
                            # obj.attrib['value'].encode('unicode-escape').decode('utf-8')
                            pv = Helper.remove_emojis(obj.attrib['value'])
                        except Exception as e:
                            #print('Error:', e)
                            pass
                        if (pv != ''):
                            continue
                        objdict = obj[0].attrib

                        if ('id' in objdict):
                            #print('--', objdict['id'])
                            ob = Helper.remove_emojis(objdict['id']) # qualifier
                        elif ('time' in objdict):
                            #print('--', objdict['time'])
                            pv = Helper.remove_emojis(objdict['time']) # time
                        elif ('amount' in objdict):
                            #print('--', objdict['amount'])
                            pv = Helper.remove_emojis(objdict['amount']) # amount
                        # capture other data types (globle coordinate, etc)
                        # ...
                        else:
                            pass
                            #print('--', 'empty')
                    except Exception as e:
                        pass
                        #print('Error:', e) 
                            
                # check the number of qualifiers
                qual_properties = [t for t in claim.findall('qualifiers/property')]
                if (len(qual_properties) == 0):
                    if (pt != 'wikibase-item'):
                        r1 = [s, p, pt, pv]
                        claim_list.append(['r1', r1]) # WST-1 statement
                    else:
                        r2 = [s, p, pt, ob]
                        claim_list.append(['r2', r2]) # WST-2 statement
                else:
                    if (pv != ''):
                        r3 = [s, p, pt, pv] # WST3-a
                    else:
                        r3 = [s, p, pt, ob] # WST3-b 
                    try:
                        for x in claim.find('qualifiers'):
                            #print('----', x.attrib['id'], x.tag)
                            q = Helper.remove_emojis(x.attrib['id']) # qualifier identifier
                            qt = Helper.remove_emojis(x[0].attrib['datatype']) # qualifier data type
                            subr = [q, qt]
                            #children = x.getchildren()
                            children =  list(x)
                            for y in children:
                                for z in y.find('datavalue'):
                                    qv = '' # qualifier value
                                    if ('id' in z.attrib):
                                        #print('--------', z.attrib['id'])
                                        qv = Helper.remove_emojis(z.attrib['id']) # qualifier value
                                    elif ('time' in z.attrib):
                                        #print('--------', z.attrib['time'])
                                        qv = Helper.remove_emojis(z.attrib['time']) # value
                                    elif ('amount' in z.attrib):
                                        #print('--------', z.attrib['amount'])
                                        qv = Helper.remove_emojis(z.attrib['amount']) # value
                                    # capture other data types (globle coordinate, etc)
                                    # ...   
                                    else:
                                        #print('--------', 'empty')
                                        qv = '' # set to empty
                                    if (qv != ''):
                                        subr.append(qv)
                                        r3.append(subr) # add a qualifier value
                                        qv = '' # set to empty for new iterator
                    except Exception as e:
                        pass
                        #print('Error: ', e)
                    
                    if (len(r3) > 4):
                        claim_list.append(['r3', r3]) # WST-3 statement
                    else:
                        if (pt != 'wikibase-item'):
                            claim_list.append(['r1', r3]) # WST-1 statement
                        else:
                            claim_list.append(['r2', r3]) # WST-2 statement
                ob = pv = '' # reset values (important)
            #print('************************')    
        return claim_list     

    def filter_claim_by_type(self, claim_list, claim_type = 'r1'):
        """
            filter claims by types
                claim_list: list - a list of claims
                claim_type: string - claim type (r1 = relation type 1, r2 = relation type 2, r3 =  relation type 3)
                return: list - a filtered claim lists
        """
        
        result_list = []
        for x in claim_list:
            if (x[0] == claim_type): result_list.append(x)
               
        return result_list     

    def get_property_by_object_id(self, object_id, claim_list):
        """
            get properies by an object id
                object_id: string - an identifier of an object
                claims: list - a list of claims
                return: list - a property list
        """
        
        property_list = []
        for x in claim_list:
            if (x[1][2] == object_id):
                for y in range(4, len(x[1])):
                    property_list.append(x[1][y])
        
        return property_list   

    def get_hypernyms(self, values, results, level = 3, language = 'en'): # error
        """
            get Wikidata hypernyms by hypernym level (recursive)
                values: list - a given list
                results: list - a list of hypernyms
                level: int - the level of hypernym
                return: list - a list of hypernyms
        """

        if (level == 0 or len(values) == 0):
            results += values
            results = [list(x) for x in set(tuple(x) for x in results)] # keep unique values
            return results

        terms = []
        for v in values:
            try:
                root = self.get_wikidata_root(v[0])
                claim_list = self.get_claim(root, v[0])

                terms += self.get_value_by_property(claim_list, 'P31', language)
                terms += self.get_value_by_property(claim_list, 'P279', language)
            except: pass
            
        results += values
        results += terms
        results = [list(x) for x in set(tuple(x) for x in results)] # keep unique values

        terms = set(tuple(x) for x in terms)
        values = set(tuple(x) for x in values)
        terms = terms - values
        terms  = [list(x) for x in terms]

        return self.get_hypernyms(terms, results, level - 1, language)
    
    def get_triple(self, word, level = 2):

        """
            English only
        """
        
        triple_list = []
        term_dict = self.get_item_by_title(word)
        if not term_dict: return []
    
        instances = term_dict['instances']
        subclasses = term_dict['subclasses']
        parts = term_dict['parts']
            
        for i in instances: triple_list.append([term_dict['label'], 'instance_of', i[1]])
        for s in subclasses: triple_list.append([term_dict['label'], 'subclass_of', s[1]])
        for p in parts: triple_list.append([term_dict['label'], 'part_of', p[1]])

        # get property data
        triple_list += self.get_extend_triple(triple_list, instances, subclasses, parts, level)
    
        # filter triples
        filtered_list = []
        for t in triple_list:
            flag = True
            for f in filtered_list:
                if (t[0] == f[0] and t[1] == f[1] and t[2] == f[2]):
                    flag = False
                    break
            if (flag == True): filtered_list.append(t)

        return filtered_list

    def get_extend_triple(self, triple_list, instances, subclasses, parts, level):

        """
            English only
        """

        if (level == 0): return triple_list
    
        # get property data
        id_list = [i[0] for i in instances]
        id_list += [s[0] for s in subclasses]
        id_list += [p[0] for p in parts]

        for item in id_list:
            term_dict = self.get_item_by_id(item)
            try: 
                instances = term_dict['instances']
                for i in instances: triple_list.append([term_dict['label'], 'instance_of', i[1]])
            except Exception as e:
                #print('Error1 -- get_extend_triple: ', e)
                instances = {}
                pass

            try:
                subclasses = term_dict['subclasses']
                for s in subclasses: triple_list.append([term_dict['label'], 'subclass_of', s[1]])
            except Exception as e:
                #print('Error2 -- get_extend_triple: ', e)
                subclasses = {}
                pass

            try:
                parts = term_dict['parts']
                for p in parts: triple_list.append([term_dict['label'], 'part_of', p[1]])
            except Exception as e:
                #print('Error3 -- get_extend_triple: ', e)
                parts = {}
                pass

        level = level - 1
        #print(triple_list, instances, subclasses, parts, level)
    
        return self.get_extend_triple(triple_list, instances, subclasses, parts, level)
        
    def search_wikidata(self, term, search_type = 'property', limit = 50, data_format = 'xml'):

        result_list = []
        link = 'https://www.wikidata.org/w/api.php'
        params = {
            'action': 'wbsearchentities',
            'language': 'en',
            'format': data_format,
            'type': search_type,
            'limit': limit,
            'search': term
            }

        print(Helper.create_link(link, params))
        #print('params: ', params)
        
        #response = requests.get(link, params)
        response = WikiRequest.get_data_by_link(link, params)
        
        for x in response.find('./search'):
            try:
                wdid = Helper.remove_emojis(x.attrib['id'])
                title = Helper.remove_emojis(x.attrib['title'])
                label = Helper.remove_emojis(x.attrib['label'])
                object_type = ''

                if ('Property:' in title):
                    datatype = Helper.remove_emojis(x.attrib['datatype'])
                    object_type = 'property'
                else:
                    datatype = 'wikibase-item'
                    object_type = 'item'
                #text = Helper.remove_emojis(x.attrib['text'])
                if (wdid != ''):
                    result_list.append({'wikidata_id': wdid, 'title': title , 'label': label, 'datatype': datatype, 'object_type': object_type})
            except:
                pass
                
        return result_list
# -----------------------------------------------------------

# class Page ------------------------------------------------
class Page():
    def __init__(self, idx = 0, namespace = '', input_title = '', title = '', content = '', first_paragraph = '', \
                 first_sentence = '', categories = [], templates = []):

        self.idx = idx
        self.namespace = namespace
        self.input_title = input_title
        self.title = title
        self.content = content
        self.first_paragraph = first_paragraph
        self.first_sentence = first_sentence
        self.categories = categories
        self.templates = templates
        
# -----------------------------------------------------------

# class Wikipedia -------------------------------------------
class Wikipedia():

    def get_page(self, title, data_format = 'json', language = 'en', redirect = True, return_type = 'dict',  timeout = 45):

        response = self.get_data_by_title(title, data_format, language, redirect, timeout)
        #print('response: ', response)
        
        page_dict = {}
        page_dict['idx'] = self.get_idx(response, data_format)
        page_dict['namespace'] = self.get_namespace(response, data_format)
        page_dict['input_title'] = title # in case of redirects
        page_dict['title'] = self.get_title(response, data_format)
        page_dict['content'] = self.get_page_content(response, data_format)
        page_dict['raw_content'] = self.get_raw_page_content(response, data_format)
        page_dict['first_paragraph'] = self.get_first_paragraph(page_dict['content'])
        page_dict['first_raw_paragraph'] = self.get_first_raw_paragraph(page_dict['raw_content'])
        page_dict['first_sentence'] = self.get_first_sentence(page_dict['first_paragraph'])
        page_dict['categories'] = self.get_category(response, data_format)
        page_dict['templates'] = self.get_template(response, data_format)

        if (return_type == 'class'):
            return Page(page_dict['idx'], page_dict['namespace'], page_dict['input_title'], page_dict['title'], \
                        page_dict['content'], page_dict['first_paragraph'], page_dict['first_sentence'], \
                        page_dict['categories'], page_dict['templates'])

        #print('page_dict: ', page_dict['content'])
        return page_dict

    def get_title(self, data, data_format):
        """
            get title
                data: string - data
                data_format: string - XML or JSON
                return: string - title
        """
        
        title = ''
        if (data_format == 'xml'):
            for node in data.find('./query/pages/page'):
                title = node.attrib['title']
        elif(data_format == 'json'):
            try:
                data_dict = data['query']['pages']
                key = next(iter(data_dict))
                title = data_dict[key]['title']
            except Exception as e:
                #print('Error --- get_title: ', e)
                pass
            
        return title
    
    def get_namespace(self, data, data_format):
        """
            get namespace
                data: string - data
                data_format: string - XML or JSON
                return: int - namespace (https://en.wikipedia.org/wiki/Wikipedia:Namespace)
        """
        
        ns = ''
        if (data_format == 'xml'):
            for node in data.find('./query/pages/page'):
                ns = node.attrib['ns']
        elif(data_format == 'json'):
            try:
                data_dict = data['query']['pages']
                key = next(iter(data_dict))
                ns = data_dict[key]['ns']
            except Exception as e:
                #print('Error --- get_namespace: ', e)
                pass
            
        return ns

    def get_idx(self, data, data_format):
        """
            get Wikidata identifier
                data: string - data
                data_format: string - XML or JSON
                return: string - Wikidata identifier
        """
        
        idx = ''
        if (data_format == 'xml'):
            for node in data.find('./query/pages/page'):
                idx = node.attrib['pageid']
        elif(data_format == 'json'):
            try:
                data_dict = data['query']['pages']
                key = next(iter(data_dict))
                idx = data_dict[key]['pageid']
            except Exception as e:
                #print('Error --- get_idx: ', e)
                pass
            
        return idx
    

    def get_category(self, data, data_format):
        """
            get categories
                data: string - data
                data_format: string - XML or JSON
                return: list - list of categories
        """
        
        cat_list = []
        if (data_format == 'xml'):
            for node in data.find('./query/pages/page/categories/cl'):
                cat_list.append(node.text)
        elif(data_format == 'json'):
            try:
                data_dict = data['query']['pages']
                key = next(iter(data_dict))
                data_list = data_dict[key]['categories']

                for item_dict in data_list:
                    if (item_dict['ns'] != 14): continue
                    cat_list.append(item_dict['title'])
                    
            except Exception as e:
                #print('Error --- get_category: ', e)
                pass

        return cat_list

    def get_template(self, data, data_format):
        """
            get templates
                data: string - data
                data_format: string - XML or JSON
                return: list - list of templates
        """
        
        temp_list = []
        if (data_format == 'xml'):
            for node in data.find('./query/pages/page/templates/tl'):
                temp_list.append(node.text)
        elif(data_format == 'json'):
            try:
                data_dict = data['query']['pages']
                key = next(iter(data_dict))
                data_list = data_dict[key]['templates']

                for item_dict in data_list:
                    if (item_dict['ns'] != 10): continue
                    temp_list.append(item_dict['title'])
                    
            except Exception as e:
                #print('Error --- get_template: ', e)
                pass

        #cat_list = [Helper.remove_emojis(c) for c in cat_list]
        return temp_list
        

    def get_data_by_title(self, title, data_format = 'xml', language = 'en', redirect = True, timeout = 45):
        """
            get data by title and language
                title: string - page title
                language: string - language
                return: string - return data
        """

        if (title == ''): return ''
    
        link = 'https://' + language + '.wikipedia.org/w/api.php'
        params = {
            'action': 'query',
            'format': data_format,
            'explaintext': True,
            'rvprop': 'content',
            'prop': 'extracts|revisions|pageprops|templates|categories',
            #'exintro': False,
            'rvprop': 'content',
            'rvslots': 'main',
            'titles': title,
            'clshow': '!hidden' # not show hidden categories
            
            }

        if (redirect == False): params['redirects'] = False

        print(Helper.create_link(link, params))
        response = WikiRequest.get_data_by_link(link, params, data_format, timeout)
        return response
    
    def get_first_paragraph(self, content):
        """
            get first paragraph
                content: page content - string
                return: string 
        """

        first_paragraph = ''
        try:
            first_paragraph = content[0:content.index('==')].strip('\n')
        except:
            first_paragraph = content.strip()
            pass

        return first_paragraph


    def get_raw_page_content(self, data, data_format):
        """
            get page content
                data: string - data
                data_format: string - XML or JSON
                return: string - content
        """

        content = ''
        if (data_format  == 'xml'):
            for node in data.find('./query/pages/page'):
                if (node.tag == 'extract'):
                    content = node.text
                    break
        elif(data_format == 'json'):
            try:
                data_dict = data['query']['pages']
                key = next(iter(data_dict))
                content = data_dict[key]['revisions'][0]['slots']['main']['*']
        
            except Exception as e:
                #print('Error --- get_page_content: ', e)
                pass

        content = Helper.remove_emojis(content)
        #print('content: ', content)
        return content

    def get_first_raw_paragraph(self, content):
        """
            get first paragraph
                content: page content - string
                return: string 
        """

        first_paragraph = ''
        try:
            first_paragraph = content[0:content.index('==')].strip('\n')
        except:
            pass

        return first_paragraph
  
    
    def get_page_content(self, data, data_format):
        """
            get page content
                data: string - data
                data_format: string - XML or JSON
                return: string - content
        """

        content = ''
        if (data_format  == 'xml'):
            for node in data.find('./query/pages/page'):
                if (node.tag == 'extract'):
                    content = node.text
                    break
        elif(data_format == 'json'):
            try:
                data_dict = data['query']['pages']
                key = next(iter(data_dict))
                content = data_dict[key]['extract']
        
            except Exception as e:
                #print('Error --- get_page_content: ', e)
                pass

        #content = Helper.remove_emojis(content)
        return content
    
    def get_sentence_list(self, text, tool = 'spacy'):
        """
            get sentence list from text
                text: string - a given text
                return: list - a list of sentences
        """
    
        #text = Helper.remove_emojis(text)
        sen_list = []
        
        if (tool == 'spacy'):
            doc = nlp(text)
            for sent in doc.sents:
                if ('wikivoyage' in sent.text.lower()): continue
                if ('wikimedia' in sent.text.lower()): continue
                if ('wikipedia' in sent.text.lower()): continue
                if ('wiktionary' in sent.text.lower()): continue
                if ('.' not in sent.text): continue
                
                sent_strip = ' '.join(x.strip() for x in sent.text.split() if x.strip() != '')
                sen_list.append(sent_strip)
        else: # nlkt punkt
            for sent in sent_detector.tokenize(text):
                if ('wikivoyage' in sent.lower()): continue
                if ('wikimedia' in sent.lower()): continue
                if ('wikipedia' in sent.lower()): continue
                if ('wiktionary' in sent.lower()): continue
                if ('.' not in sent): continue
                
                sent_strip = ' '.join(x.strip() for x in sent.split() if x.strip() != '')
                sen_list.append(sent_strip)
            
            
        sen_list = [x.strip() for x in sen_list if x.strip() != '' and '==' not in x]
        return sen_list

    def get_first_sentence(self, text):

        try:
            return self.get_sentence_list(text)[0]
        except:
            return ''

    def get_content_by_section(self, content):
        """
            split content into sections
                content: string - content
                return: dict - content dict
        """
    
        return
    
    def search_wikipedia(self, term, limit = 1, language = 'en', data_format = 'json', key_len = 2, timeout = 45):

        result_term = ''
        suggestion = False

        if (len(term) < key_len): return {'value': result_term, 'suggestion': suggestion}
        
        link = 'https://' +  language + '.wikipedia.org/w/api.php'
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': term,
            'srlimit': limit,
            'srprop': 'size',
            'formatversion': 2,
            'format': data_format
            }

        #response = requests.get(link, params)
        response = WikiRequest.get_data_by_link(link, params, data_format, timeout)
        
        if (data_format == 'json'):
            try:
                result_term = response['query']['searchinfo']['suggestion']
                suggestion = True
                if (term.strip().lower() == result_term.strip().lower()): result_term = term
                
            except:
                try:
                    result_term = response['query']['search'][0]['title']
                except:
                    #print('Error1 --- search_wikipedia: ', e)
                    pass
                #print('Error2 --- search_wikipedia: ', e)
                pass
        elif (data_format == 'xml'):
            # define later...
            pass

        return {'value': result_term, 'suggestion': suggestion}

# ---------------------------------------------------------------

# class WikidataQuery -------------------------------------------
class WikidataQuery():

    @staticmethod
    def get_data(query, data_format = 'json', timeout = 60):
        """
            get the data from Wikidata server by SPARQL queries
                query: string - a SPARQL query
                data_format: string - the format of return data
                return: list - a list of results
        """

        link = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
        params = {
            'query': query,
            'format': data_format
            }
        
        #response = requests.get(link, params)
        response = WikiRequest.get_data_by_link(link, params, data_format, timeout)
        
        result_list = []
        if (data_format == 'xml'):
            '''
            for x in root.find('./results/result'):
                try:
                    name = Helper.remove_emojis(x.attrib['name'])
                    value = Helper.remove_emojis(x.find('uri').value)
                except:
                    pass
            '''
            pass
        
        elif(data_format == 'json'):
            #root = json.loads(response.text)
            bindings = response['results']['bindings']
            for binding in bindings:
                result_list.append(binding)

        return result_list
# -----------------------------------------------------------

# -----------------------------------------------------------
#if __name__ == "__main__":

    #api = Wikidata()
    #print(api.get_item_by_id('Q7076854'))

    #api = Wikidata()
    #print(api.search_wikidata('Republic of China', search_type = 'item', limit = 5))


    




    


    


