#!/usr/local/bin/python

"""Inter-annotator agreement calculator."""

"""
To run this file, please use:

python <gold standard folder> <system output folder>

e.g.: python gold_annotations system_annotations

Please note that you must use Python 3 to get the correct results with this script


"""

import argparse
import glob
import os
import logging
from collections import defaultdict
from xml.etree import cElementTree

logger = logging.getLogger(__name__)


class ClinicalCriteria(object):
    """Criteria in the Track 1 documents."""

    def __init__(self, tid, value):
        """Init."""
        self.tid = tid.strip().upper()
        self.ttype = self.tid
        self.value = value.lower().strip()

    def equals(self, other, mode='strict'):
        """Return whether the current criteria is equal to the one provided."""
        if other.tid == self.tid and other.value == self.value:
            return True
        return False


class ClinicalConcept(object):
    """Named Entity Tag class."""

    def __init__(self, tid, start, end, ttype, text=''):
        """Init."""
        self.tid = str(tid).strip()
        self.start = int(start)
        self.end = int(end)
        self.text = str(text).strip()
        self.ttype = str(ttype).strip()

    def span_matches(self, other, mode='strict'):
        """Return whether the current tag overlaps with the one provided."""
        assert mode in ('strict', 'lenient')
        if mode == 'strict':
            if self.start == other.start and self.end == other.end:
                return True
        else:  # lenient
            if (self.end > other.start and self.start < other.end) or \
                    (self.start < other.end and other.start < self.end):
                return True
        return False

    def equals(self, other, mode='strict'):
        """Return whether the current tag is equal to the one provided."""
        assert mode in ('strict', 'lenient')
        return other.ttype == self.ttype and self.span_matches(other, mode)

    def __str__(self):
        """String representation."""
        return '{}\t{}\t({}:{})'.format(self.ttype, self.text, self.start, self.end)


class Relation(object):
    """Relation class."""

    def __init__(self, rid, arg1, arg2, rtype):
        """Init."""
        assert isinstance(arg1, ClinicalConcept)
        assert isinstance(arg2, ClinicalConcept)
        self.rid = str(rid).strip()
        self.arg1 = arg1
        self.arg2 = arg2
        self.rtype = str(rtype).strip()

    def equals(self, other, mode='strict'):
        """Return whether the current tag is equal to the one provided."""
        assert mode in ('strict', 'lenient')
        if self.arg1.equals(other.arg1, mode) and \
                self.arg2.equals(other.arg2, mode) and \
                self.rtype == other.rtype:
            return True
        return False

    def __str__(self):
        """String representation."""
        return '{} ({}->{})'.format(self.rtype, self.arg1.ttype,
                                    self.arg2.ttype)


class RecordTrack1(object):
    """Record for Track 2 class."""

    def __init__(self, file_path):
        self.path = os.path.abspath(file_path)
        self.basename = os.path.basename(self.path)
        self.annotations = self._get_annotations()
        self.text = None

    @property
    def tags(self):
        return self.annotations['tags']

    def _get_annotations(self):
        """Return a dictionary with all the annotations in the .ann file."""
        annotations = defaultdict(dict)
        annotation_file = cElementTree.parse(self.path)
        for tag in annotation_file.findall('.//TAGS/*'):
            criterion = ClinicalCriteria(tag.tag.upper(), tag.attrib['met'])
            annotations['tags'][tag.tag.upper()] = criterion
            if tag.attrib['met'] not in ('met', 'not met'):
                assert '{}: Unexpected value ("{}") for the {} tag!'.format(
                    self.path, criterion.value, criterion.ttype)
        return annotations


class RecordTrack2(object):
    """Record for Track 2 class."""

    def __init__(self, file_path):
        """Initialize."""
        self.path = os.path.abspath(file_path)
        self.basename = os.path.basename(self.path)
        self.annotations = self._get_annotations()
        # self.text = self._get_text()

    @property
    def tags(self):
        return self.annotations['tags']

    @property
    def relations(self):
        return self.annotations['relations']

    def _get_annotations(self):
        """Return a dictionary with all the annotations in the .ann file."""
        annotations = defaultdict(dict)
        with open(self.path) as annotation_file:
            lines = annotation_file.readlines()
            for line_num, line in enumerate(lines):
                if line.strip().startswith('T'):
                    try:
                        tag_id, tag_m, tag_text = line.strip().split('\t')
                    except ValueError:
                        print(self.path, line)
                    if len(tag_m.split(' ')) == 3:
                        tag_type, tag_start, tag_end = tag_m.split(' ')
                    elif len(tag_m.split(' ')) == 4:
                        tag_type, tag_start, _, tag_end = tag_m.split(' ')
                    elif len(tag_m.split(' ')) == 5:
                        tag_type, tag_start, _, _, tag_end = tag_m.split(' ')
                    else:
                        print(self.path)
                        print(line)
                    tag_start, tag_end = int(tag_start), int(tag_end)
                    annotations['tags'][tag_id] = ClinicalConcept(tag_id,
                                                                  tag_start,
                                                                  tag_end,
                                                                  tag_type,
                                                                  tag_text)
            for line_num, line in enumerate(lines):
                if line.strip().startswith('R'):
                    rel_id, rel_m = line.strip().split('\t')
                    rel_type, rel_arg1, rel_arg2 = rel_m.split(' ')
                    rel_arg1 = rel_arg1.split(':')[1]
                    rel_arg2 = rel_arg2.split(':')[1]
                    try:
                        arg1 = annotations['tags'][rel_arg1]
                        arg2 = annotations['tags'][rel_arg2]
                        annotations['relations'][rel_id] = Relation(rel_id, arg1,
                                                                    arg2, rel_type)
                    except KeyError as err:
                        logger.info(err)
        return annotations

    def _get_text(self):
        """Return the text in the corresponding txt file."""
        path = self.path.replace('.ann', '.txt')
        with open(path) as text_file:
            text = text_file.read()
        return text

    def search_by_id(self, key):
        """Search by id among both tags and relations."""
        try:
            return self.annotations['tags'][key]
        except KeyError():
            try:
                return self.annotations['relations'][key]
            except KeyError():
                return None


class Measures(object):
    """Abstract methods and var to evaluate."""

    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        """Initizialize."""
        assert type(tp) == int
        assert type(tn) == int
        assert type(fp) == int
        assert type(fn) == int
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def precision(self):
        """Compute Precision score."""
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return 0.0

    def recall(self):
        """Compute Recall score."""
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return 0.0

    def f_score(self, beta=1):
        """Compute F1-measure score."""
        assert beta > 0.
        try:
            num = (1 + beta ** 2) * (self.precision() * self.recall())
            den = beta ** 2 * (self.precision() + self.recall())
            return num / den
        except ZeroDivisionError:
            return 0.0

    def f1(self):
        """Compute the F1-score (beta=1)."""
        return self.f_score(beta=1)

    def specificity(self):
        """Compute Specificity score."""
        try:
            return self.tn / (self.fp + self.tn)
        except ZeroDivisionError:
            return 0.0

    def sensitivity(self):
        """Compute Sensitivity score."""
        return self.recall()

    def auc(self):
        """Compute AUC score."""
        return (self.sensitivity() + self.specificity()) / 2


class SingleEvaluator(object):
    """Evaluate two single files."""

    def __init__(self, doc1, doc2, track, mode='strict', key=None, verbose=False, exclude_tags=()):
        """Initialize."""
        assert isinstance(doc1, RecordTrack2) or isinstance(doc1, RecordTrack1)
        assert isinstance(doc2, RecordTrack2) or isinstance(doc2, RecordTrack1)
        assert mode in ('strict', 'lenient')
        assert doc1.basename == doc2.basename
        self.scores = {'tags': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
                       'relations': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}}
        self.doc1 = doc1
        self.doc2 = doc2
        if key:
            gol = [t for t in doc1.tags.values() if t.ttype == key and t.ttype not in exclude_tags]
            sys = [t for t in doc2.tags.values() if t.ttype == key and t.ttype not in exclude_tags]
            sys_check = [t for t in doc2.tags.values() if t.ttype == key and t.ttype not in exclude_tags]
        else:
            gol = [t for t in doc1.tags.values() if t.ttype not in exclude_tags]
            sys = [t for t in doc2.tags.values() if t.ttype not in exclude_tags]
            sys_check = [t for t in doc2.tags.values() if t.ttype not in exclude_tags]

        # pare down matches -- if multiple system tags overlap with only one
        # gold standard tag, only keep one sys tag
        gol_matched = []
        for s in sys:
            for g in gol:
                if (g.equals(s, mode)):
                    if g not in gol_matched:
                        gol_matched.append(g)
                    else:
                        if s in sys_check:
                            sys_check.remove(s)

        sys = sys_check
        # now evaluate
        self.scores['tags']['tp'] = len({s.tid for s in sys for g in gol if g.equals(s, mode)})
        self.scores['tags']['fp'] = len({s.tid for s in sys}) - self.scores['tags']['tp']
        self.scores['tags']['fn'] = len({g.tid for g in gol}) - self.scores['tags']['tp']
        self.scores['tags']['tn'] = 0

        if verbose and track == 2:
            tps = {s for s in sys for g in gol if g.equals(s, mode)}
            fps = set(sys) - tps
            fns = set()
            for g in gol:
                if not len([s for s in sys if s.equals(g, mode)]):
                    fns.add(g)
            for e in fps:
                print('FP: ' + str(e))
            for e in fns:
                print('FN:' + str(e))
        if track == 2:
            if key:
                gol = [r for r in doc1.relations.values() if r.rtype == key]
                sys = [r for r in doc2.relations.values() if r.rtype == key]
                sys_check = [r for r in doc2.relations.values() if r.rtype == key]
            else:
                gol = [r for r in doc1.relations.values()]
                sys = [r for r in doc2.relations.values()]
                sys_check = [r for r in doc2.relations.values()]

            # pare down matches -- if multiple system tags overlap with only one
            # gold standard tag, only keep one sys tag
            gol_matched = []
            for s in sys:
                for g in gol:
                    if (g.equals(s, mode)):
                        if g not in gol_matched:
                            gol_matched.append(g)
                        else:
                            if s in sys_check:
                                sys_check.remove(s)
            sys = sys_check
            # now evaluate
            self.scores['relations']['tp'] = len({s.rid for s in sys for g in gol if g.equals(s, mode)})
            self.scores['relations']['fp'] = len({s.rid for s in sys}) - self.scores['relations']['tp']
            self.scores['relations']['fn'] = len({g.rid for g in gol}) - self.scores['relations']['tp']
            self.scores['relations']['tn'] = 0
            if verbose:
                tps = {s for s in sys for g in gol if g.equals(s, mode)}
                fps = set(sys) - tps
                fns = set()
                for g in gol:
                    if not len([s for s in sys if s.equals(g, mode)]):
                        fns.add(g)
                for e in fps:
                    print('FP: ' + str(e))
                for e in fns:
                    print('FN:' + str(e))


class MultipleEvaluator(object):
    """Evaluate two sets of files."""

    def __init__(self, corpora, tag_type=None, mode='strict',
                 verbose=False):
        """Initialize."""
        assert isinstance(corpora, Corpora)
        assert mode in ('strict', 'lenient')
        self.scores = None
        if corpora.track == 1:
            self.track1(corpora)
        else:
            self.track2(corpora, tag_type, mode, verbose)

    def track1(self, corpora):
        """Compute measures for Track 1."""
        self.tags = ('ABDOMINAL', 'ADVANCED-CAD', 'ALCOHOL-ABUSE',
                     'ASP-FOR-MI', 'CREATININE', 'DIETSUPP-2MOS',
                     'DRUG-ABUSE', 'ENGLISH', 'HBA1C', 'KETO-1YR',
                     'MAJOR-DIABETES', 'MAKES-DECISIONS', 'MI-6MOS')
        self.scores = defaultdict(dict)
        metrics = ('p', 'r', 'f1', 'specificity', 'auc')
        values = ('met', 'not met')
        self.values = {'met': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
                       'not met': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}}

        def evaluation(corpora, value, scores):
            predictions = defaultdict(list)
            for g, s in corpora.docs:
                for tag in self.tags:
                    predictions[tag].append(
                        (g.tags[tag].value == value, s.tags[tag].value == value))
            for tag in self.tags:
                # accumulate for micro overall measure
                self.values[value]['tp'] += predictions[tag].count((True, True))
                self.values[value]['fp'] += predictions[tag].count((False, True))
                self.values[value]['tn'] += predictions[tag].count((False, False))
                self.values[value]['fn'] += predictions[tag].count((True, False))

                # compute per-tag measures
                measures = Measures(tp=predictions[tag].count((True, True)),
                                    fp=predictions[tag].count((False, True)),
                                    tn=predictions[tag].count((False, False)),
                                    fn=predictions[tag].count((True, False)))
                scores[(tag, value, 'p')] = measures.precision()
                scores[(tag, value, 'r')] = measures.recall()
                scores[(tag, value, 'f1')] = measures.f1()
                scores[(tag, value, 'specificity')] = measures.specificity()
                scores[(tag, value, 'auc')] = measures.auc()
            return scores

        self.scores = evaluation(corpora, 'met', self.scores)
        self.scores = evaluation(corpora, 'not met', self.scores)

        for measure in metrics:
            for value in values:
                self.scores[('macro', value, measure)] = sum(
                    [self.scores[(t, value, measure)] for t in self.tags]) / len(self.tags)

    def track2(self, corpora, tag_type=None, mode='strict', verbose=False):
        """Compute measures for Track 2."""
        self.scores = {'tags': {'tp': 0,
                                'fp': 0,
                                'fn': 0,
                                'tn': 0,
                                'micro': {'precision': 0,
                                          'recall': 0,
                                          'f1': 0},
                                'macro': {'precision': 0,
                                          'recall': 0,
                                          'f1': 0}},
                       'relations': {'tp': 0,
                                     'fp': 0,
                                     'fn': 0,
                                     'tn': 0,
                                     'micro': {'precision': 0,
                                               'recall': 0,
                                               'f1': 0},
                                     'macro': {'precision': 0,
                                               'recall': 0,
                                               'f1': 0}}}

        # -----------------------------------------CG--------------------------------------------
        if corpora.corpus_type == 'cg':
            self.tags = (
                'Gene_expression', 'Mutation', 'Regulation', 'Development', 'Negative_regulation', 'Cell_proliferation',
                'Transcription', 'Glycosylation', 'Positive_regulation', 'Binding', 'Localization', 'Planned_process',
                'Metastasis', 'Death', 'Blood_vessel_development', 'Breakdown', 'Growth', 'Cell_transformation',
                'Carcinogenesis', 'Cell_differentiation', 'Cell_death', 'Cell_division', 'Infection', 'Pathway',
                'Dephosphorylation', 'Synthesis', 'Catabolism', 'Protein_processing', 'Remodeling', 'Metabolism',
                'Dissociation', 'Phosphorylation', 'Glycolysis', 'Translation', 'DNA_methylation', 'Reproduction',
                'Acetylation', 'Ubiquitination', 'Amino_acid_catabolism', 'DNA_demethylation', 'Gene_or_gene_product',
                'Cancer', 'Cell', 'Organism', 'DNA_domain_or_region', 'Simple_chemical', 'Multi-tissue_structure',
                'Organ', 'Organism_subdivision', 'Tissue', 'Immaterial_anatomical_entity', 'Organism_substance',
                'Protein_domain_or_region', 'Cellular_component', 'Pathological_formation', 'Amino_acid',
                'Anatomical_system', 'Developing_anatomical_structure')
            self.exclude_tags = ()
            self.relations = (
                'CSite', 'FromLoc', 'Site', 'ToLoc', 'Participant', 'AtLoc', 'Instrument', 'Cause', 'Theme')
        elif corpora.corpus_type == 'cg_tr':
            self.tags = (
                'Gene_expression', 'Mutation', 'Regulation', 'Development', 'Negative_regulation', 'Cell_proliferation',
                'Transcription', 'Glycosylation', 'Positive_regulation', 'Binding', 'Localization', 'Planned_process',
                'Metastasis', 'Death', 'Blood_vessel_development', 'Breakdown', 'Growth', 'Cell_transformation',
                'Carcinogenesis', 'Cell_differentiation', 'Cell_death', 'Cell_division', 'Infection', 'Pathway',
                'Dephosphorylation', 'Synthesis', 'Catabolism', 'Protein_processing', 'Remodeling', 'Metabolism',
                'Dissociation', 'Phosphorylation', 'Glycolysis', 'Translation', 'DNA_methylation', 'Reproduction',
                'Acetylation', 'Ubiquitination', 'Amino_acid_catabolism', 'DNA_demethylation', 'Gene_or_gene_product',
                'Cancer', 'Cell', 'Organism', 'DNA_domain_or_region', 'Simple_chemical', 'Multi-tissue_structure',
                'Organ', 'Organism_subdivision', 'Tissue', 'Immaterial_anatomical_entity', 'Organism_substance',
                'Protein_domain_or_region', 'Cellular_component', 'Pathological_formation', 'Amino_acid',
                'Anatomical_system', 'Developing_anatomical_structure')
            self.exclude_tags = (
                'Immaterial_anatomical_entity', 'Cancer', 'Multi-tissue_structure', 'Anatomical_system',
                'Pathological_formation', 'Tissue', 'Gene_or_gene_product', 'Cell', 'Protein_domain_or_region',
                'Developing_anatomical_structure', 'Organism', 'Organ', 'Simple_chemical', 'Organism_subdivision',
                'Amino_acid', 'Organism_substance', 'DNA_domain_or_region', 'Cellular_component'
            )
            self.relations = (
                'CSite', 'FromLoc', 'Site', 'ToLoc', 'Participant', 'AtLoc', 'Instrument', 'Cause', 'Theme'
            )
        elif corpora.corpus_type == 'cg_en':
            self.tags = (
                'Gene_expression', 'Mutation', 'Regulation', 'Development', 'Negative_regulation', 'Cell_proliferation',
                'Transcription', 'Glycosylation', 'Positive_regulation', 'Binding', 'Localization', 'Planned_process',
                'Metastasis', 'Death', 'Blood_vessel_development', 'Breakdown', 'Growth', 'Cell_transformation',
                'Carcinogenesis', 'Cell_differentiation', 'Cell_death', 'Cell_division', 'Infection', 'Pathway',
                'Dephosphorylation', 'Synthesis', 'Catabolism', 'Protein_processing', 'Remodeling', 'Metabolism',
                'Dissociation', 'Phosphorylation', 'Glycolysis', 'Translation', 'DNA_methylation', 'Reproduction',
                'Acetylation', 'Ubiquitination', 'Amino_acid_catabolism', 'DNA_demethylation', 'Gene_or_gene_product',
                'Cancer', 'Cell', 'Organism', 'DNA_domain_or_region', 'Simple_chemical', 'Multi-tissue_structure',
                'Organ', 'Organism_subdivision', 'Tissue', 'Immaterial_anatomical_entity', 'Organism_substance',
                'Protein_domain_or_region', 'Cellular_component', 'Pathological_formation', 'Amino_acid',
                'Anatomical_system', 'Developing_anatomical_structure')
            self.exclude_tags = (
                'Positive_regulation', 'Negative_regulation', 'Regulation', 'Planned_process', 'Gene_expression',
                'Localization', 'Blood_vessel_development', 'Metastasis', 'Development', 'Cell_proliferation',
                'Cell_death', 'Binding', 'Pathway', 'Mutation', 'Cell_transformation', 'Carcinogenesis',
                'Growth', 'Death', 'Transcription', 'Breakdown', 'Cell_differentiation', 'Phosphorylation',
                'Metabolism', 'Glycolysis', 'Synthesis', 'Remodeling', 'DNA_methylation', 'Catabolism',
                'Infection', 'Protein_processing', 'Translation', 'Glycosylation', 'Dephosphorylation',
                'Acetylation', 'Dissociation', 'Cell_division', 'Amino_acid_catabolism', 'Reproduction',
                'Ubiquitination', 'DNA_demethylation')
            self.relations = (
                'CSite', 'FromLoc', 'Site', 'ToLoc', 'Participant', 'AtLoc', 'Instrument', 'Cause', 'Theme')

        # -----------------------------------------ACE--------------------------------------------
        elif corpora.corpus_type == 'ace':
            self.tags = (
                'Die', 'Injure', 'Attack', 'Transport', 'Start-Position', 'Arrest-Jail', 'Meet', 'Transfer-Money',
                'Sue', 'Charge-Indict', 'Sentence', 'Convict', 'End-Position', 'Transfer-Ownership', 'Demonstrate',
                'Execute', 'Appeal', 'Phone-Write', 'Elect', 'Trial-Hearing', 'Release-Parole', 'Acquit', 'Fine',
                'Start-Org', 'End-Org', 'Marry', 'Declare-Bankruptcy', 'Be-Born', 'Divorce', 'Extradite', 'Pardon',
                'Nominate', 'Merge-Org', 'LOC', 'FAC', 'PER', 'ORG', 'GPE', 'Time', 'WEA', 'VEH', 'Money', 'Crime',
                'Percent', 'Job-Title')
            self.exclude_tags = ()
            self.relations = (
                'Price', 'Time-At-End', 'Time-At-Beginning', 'Time-Ending', 'Time-Before', 'Time-After', 'Prosecutor',
                'Beneficiary', 'Seller', 'Time-Starting', 'Time-Holds', 'Plaintiff', 'Sentence', 'Vehicle', 'Money',
                'Buyer', 'Adjudicator', 'Org', 'Giver', 'Position', 'Recipient', 'Origin', 'Crime', 'Instrument',
                'Defendant', 'Agent', 'Target', 'Destination', 'Attacker', 'Victim', 'Artifact', 'Person',
                'Time-Within', 'Entity', 'Place')

        elif corpora.corpus_type == 'ace_tr':
            self.tags = (
                'Die', 'Injure', 'Attack', 'Transport', 'Start-Position', 'Arrest-Jail', 'Meet', 'Transfer-Money',
                'Sue', 'Charge-Indict', 'Sentence', 'Convict', 'End-Position', 'Transfer-Ownership', 'Demonstrate',
                'Execute', 'Appeal', 'Phone-Write', 'Elect', 'Trial-Hearing', 'Release-Parole', 'Acquit', 'Fine',
                'Start-Org', 'End-Org', 'Marry', 'Declare-Bankruptcy', 'Be-Born', 'Divorce', 'Extradite', 'Pardon',
                'Nominate', 'Merge-Org', 'LOC', 'FAC', 'PER', 'ORG', 'GPE', 'Time', 'WEA', 'VEH', 'Money', 'Crime',
                'Percent', 'Job-Title')
            self.exclude_tags = (
                'ORG', 'VEH', 'Time', 'GPE', 'FAC', 'Money', 'LOC', 'PER', 'WEA', 'Job-Title', 'Percent', 'Crime'
            )
            self.relations = (
                'Price', 'Time-At-End', 'Time-At-Beginning', 'Time-Ending', 'Time-Before', 'Time-After', 'Prosecutor',
                'Beneficiary', 'Seller', 'Time-Starting', 'Time-Holds', 'Plaintiff', 'Sentence', 'Vehicle', 'Money',
                'Buyer', 'Adjudicator', 'Org', 'Giver', 'Position', 'Recipient', 'Origin', 'Crime', 'Instrument',
                'Defendant', 'Agent', 'Target', 'Destination', 'Attacker', 'Victim', 'Artifact', 'Person',
                'Time-Within', 'Entity', 'Place')

        # -----------------------------------------GE13--------------------------------------------
        elif corpora.corpus_type == 'ge13':
            self.tags = (
                "Anaphora",
                "Entity",
                "Protein",
                "Acetylation",
                "Binding",
                "Deacetylation",
                "Gene_expression",
                "Localization",
                "Negative_regulation",
                "Phosphorylation",
                "Positive_regulation",
                "Protein_catabolism",
                "Protein_modification",
                "Regulation",
                "Transcription",
                "Ubiquitination",
            )
            self.exclude_tags = (
            )
            self.relations = (
                "CSite",
                "Cause",
                "Site",
                "Theme",
                "ToLoc"
            )
        elif corpora.corpus_type == 'ge13_tr':
            self.tags = (
                "Anaphora",
                "Entity",
                "Protein",
                "Acetylation",
                "Binding",
                "Deacetylation",
                "Gene_expression",
                "Localization",
                "Negative_regulation",
                "Phosphorylation",
                "Positive_regulation",
                "Protein_catabolism",
                "Protein_modification",
                "Regulation",
                "Transcription",
                "Ubiquitination",
            )
            self.exclude_tags = (
                "Anaphora",
                "Entity",
                "Protein",
            )
            self.relations = (
                "CSite",
                "Cause",
                "Site",
                "Theme",
                "ToLoc",
            )
        elif corpora.corpus_type == 'ge13_en':
            self.tags = (
                "Anaphora",
                "Entity",
                "Protein",
                "Acetylation",
                "Binding",
                "Deacetylation",
                "Gene_expression",
                "Localization",
                "Negative_regulation",
                "Phosphorylation",
                "Positive_regulation",
                "Protein_catabolism",
                "Protein_modification",
                "Regulation",
                "Transcription",
                "Ubiquitination",
            )
            self.exclude_tags = (
                "Acetylation",
                "Binding",
                "Deacetylation",
                "Gene_expression",
                "Localization",
                "Negative_regulation",
                "Phosphorylation",
                "Positive_regulation",
                "Protein_catabolism",
                "Protein_modification",
                "Regulation",
                "Transcription",
                "Ubiquitination",
            )
            self.relations = (
                "CSite",
                "Cause",
                "Site",
                "Theme",
                "ToLoc",
            )
        # -----------------------------------------GE11--------------------------------------------
        elif corpora.corpus_type == 'ge11':
            self.tags = (
                "Entity",
                "Protein",
                "Binding",
                "Gene_expression",
                "Localization",
                "Negative_regulation",
                "Phosphorylation",
                "Positive_regulation",
                "Protein_catabolism",
                "Regulation",
                "Transcription",
            )
            self.exclude_tags = (
            )
            self.relations = (
                "AtLoc",
                "CSite",
                "Cause",
                "Site",
                "Theme",
                "ToLoc",
            )
        elif corpora.corpus_type == 'ge11_tr':
            self.tags = (
                "Entity",
                "Protein",
                "Binding",
                "Gene_expression",
                "Localization",
                "Negative_regulation",
                "Phosphorylation",
                "Positive_regulation",
                "Protein_catabolism",
                "Regulation",
                "Transcription",
            )
            self.exclude_tags = (
                "Entity",
                "Protein"
            )
            self.relations = (
                "AtLoc",
                "CSite",
                "Cause",
                "Site",
                "Theme",
                "ToLoc",
            )

        elif corpora.corpus_type == 'ge11_en':
            self.tags = (
                "Entity",
                "Protein",
                "Binding",
                "Gene_expression",
                "Localization",
                "Negative_regulation",
                "Phosphorylation",
                "Positive_regulation",
                "Protein_catabolism",
                "Regulation",
                "Transcription",
            )
            self.exclude_tags = ("Binding",
                                 "Gene_expression",
                                 "Localization",
                                 "Negative_regulation",
                                 "Phosphorylation",
                                 "Positive_regulation",
                                 "Protein_catabolism",
                                 "Regulation",
                                 "Transcription",)
            self.relations = (
                "AtLoc",
                "CSite",
                "Cause",
                "Site",
                "Theme",
                "ToLoc",
            )
        # -----------------------------------------ID--------------------------------------------
        elif corpora.corpus_type == 'id':
            self.tags = (
                "Chemical",
                "Entity",
                "Organism",
                "Protein",
                "Regulon-operon",
                "Two-component-system",
                "Binding",
                "Gene_expression",
                "Localization",
                "Negative_regulation",
                "Phosphorylation",
                "Positive_regulation",
                "Process",
                "Protein_catabolism",
                "Regulation",
                "Transcription",
            )
            self.exclude_tags = (
            )
            self.relations = (
                "AtLoc",
                "CSite",
                "Cause",
                "Participant",
                "Site",
                "Theme",
                "ToLoc"
            )
        elif corpora.corpus_type == 'id_tr':
            self.tags = (
                "Chemical",
                "Entity",
                "Organism",
                "Protein",
                "Regulon-operon",
                "Two-component-system",
                "Binding",
                "Gene_expression",
                "Localization",
                "Negative_regulation",
                "Phosphorylation",
                "Positive_regulation",
                "Process",
                "Protein_catabolism",
                "Regulation",
                "Transcription",
            )
            self.exclude_tags = (
                "Chemical",
                "Entity",
                "Organism",
                "Protein",
                "Regulon-operon",
                "Two-component-system",
            )
            self.relations = (
                "AtLoc",
                "CSite",
                "Cause",
                "Participant",
                "Site",
                "Theme",
                "ToLoc",
            )
        elif corpora.corpus_type == 'id_en':
            self.tags = (
                "Binding",
                "Gene_expression",
                "Localization",
                "Negative_regulation",
                "Phosphorylation",
                "Positive_regulation",
                "Process",
                "Protein_catabolism",
                "Regulation",
                "Transcription",
            )
            self.exclude_tags = ()
            self.relations = (
                "AtLoc",
                "CSite",
                "Cause",
                "Participant",
                "Site",
                "Theme",
                "ToLoc",
            )

        # -----------------------------------------MLEE--------------------------------------------
        elif corpora.corpus_type == 'mlee':
            self.tags = (
                'Positive_regulation', 'Blood_vessel_development', 'Negative_regulation', 'Regulation',
                'Planned_process',
                'Localization', 'Development', 'Gene_expression', 'Growth', 'Binding', 'Cell_proliferation', 'Pathway',
                'Death', 'Breakdown', 'Remodeling', 'Catabolism', 'Phosphorylation', 'Transcription', 'Synthesis',
                'DNA_methylation', 'Metabolism', 'Protein_processing', 'Acetylation', 'Translation',
                'Dephosphorylation',
                'Ubiquitination', 'Gene_or_gene_product', 'Cell', 'Drug_or_compound', 'Pathological_formation',
                'Organism', 'Multi-tissue_structure',
                'Tissue', 'Organ', 'Cellular_component', 'Organism_substance', 'DNA_domain_or_region',
                'Organism_subdivision',
                'Protein_domain_or_region', 'Anatomical_system', 'Immaterial_anatomical_entity',
                'Developing_anatomical_structure'
            )
            self.exclude_tags = ()
            self.relations = (
                'Theme', 'Cause', 'Instrument', 'AtLoc', 'Participant', 'Site', 'ToLoc', 'CSite', 'FromLoc')
        elif corpora.corpus_type == 'mlee_tr':
            self.tags = (
                'Positive_regulation', 'Blood_vessel_development', 'Negative_regulation', 'Regulation',
                'Planned_process',
                'Localization', 'Development', 'Gene_expression', 'Growth', 'Binding', 'Cell_proliferation', 'Pathway',
                'Death', 'Breakdown', 'Remodeling', 'Catabolism', 'Phosphorylation', 'Transcription', 'Synthesis',
                'DNA_methylation', 'Metabolism', 'Protein_processing', 'Acetylation', 'Translation',
                'Dephosphorylation',
                'Ubiquitination', 'Gene_or_gene_product', 'Cell', 'Drug_or_compound', 'Pathological_formation',
                'Organism', 'Multi-tissue_structure',
                'Tissue', 'Organ', 'Cellular_component', 'Organism_substance', 'DNA_domain_or_region',
                'Organism_subdivision',
                'Protein_domain_or_region', 'Anatomical_system', 'Immaterial_anatomical_entity',
                'Developing_anatomical_structure'
            )
            self.exclude_tags = (
                'Gene_or_gene_product', 'Cell', 'Drug_or_compound', 'Pathological_formation', 'Organism',
                'Multi-tissue_structure',
                'Tissue', 'Organ', 'Cellular_component', 'Organism_substance', 'DNA_domain_or_region',
                'Organism_subdivision',
                'Protein_domain_or_region', 'Anatomical_system', 'Immaterial_anatomical_entity',
                'Developing_anatomical_structure'
            )
            self.relations = (
                'Theme', 'Cause', 'Instrument', 'AtLoc', 'Participant', 'Site', 'ToLoc', 'CSite', 'FromLoc')
        elif corpora.corpus_type == 'mlee_en':
            self.tags = (
                'Positive_regulation', 'Blood_vessel_development', 'Negative_regulation', 'Regulation',
                'Planned_process',
                'Localization', 'Development', 'Gene_expression', 'Growth', 'Binding', 'Cell_proliferation', 'Pathway',
                'Death', 'Breakdown', 'Remodeling', 'Catabolism', 'Phosphorylation', 'Transcription', 'Synthesis',
                'DNA_methylation', 'Metabolism', 'Protein_processing', 'Acetylation', 'Translation',
                'Dephosphorylation',
                'Ubiquitination', 'Gene_or_gene_product', 'Cell', 'Drug_or_compound', 'Pathological_formation',
                'Organism', 'Multi-tissue_structure',
                'Tissue', 'Organ', 'Cellular_component', 'Organism_substance', 'DNA_domain_or_region',
                'Organism_subdivision',
                'Protein_domain_or_region', 'Anatomical_system', 'Immaterial_anatomical_entity',
                'Developing_anatomical_structure'
            )
            self.exclude_tags = (
                'Positive_regulation', 'Blood_vessel_development', 'Negative_regulation', 'Planned_process',
                'Regulation',
                'Localization', 'Gene_expression', 'Development', 'Growth', 'Binding', 'Cell_proliferation', 'Pathway',
                'Death', 'Breakdown', 'Remodeling', 'Phosphorylation', 'Catabolism', 'Transcription', 'Synthesis',
                'DNA_methylation', 'Metabolism', 'Protein_processing', 'Dephosphorylation', 'Reproduction',
                'Acetylation',
                'Translation', 'Cell_division', 'Dissociation', 'Ubiquitination')
            self.relations = (
                'Theme', 'Cause', 'Instrument', 'AtLoc', 'Participant', 'Site', 'ToLoc', 'CSite', 'FromLoc')

        # -----------------------------------------PC--------------------------------------------
        elif corpora.corpus_type == 'pc':
            self.tags = (
                'Positive_regulation', 'Negative_regulation', 'Regulation', 'Binding', 'Pathway', 'Phosphorylation',
                'Gene_expression', 'Activation', 'Transport', 'Conversion', 'Localization', 'Inactivation',
                'Transcription',
                'Dissociation', 'Degradation', 'Ubiquitination', 'Acetylation', 'Dephosphorylation', 'Translation',
                'Methylation',
                'Demethylation', 'Deubiquitination', 'Hydroxylation', 'Deacetylation',
                'Gene_or_gene_product', 'Simple_chemical', 'Complex', 'Cellular_component'
            )
            self.exclude_tags = (
            )
            self.relations = (
                'Theme', 'Cause', 'Participant', 'Site', 'Product', 'ToLoc', 'AtLoc', 'FromLoc'
            )
        elif corpora.corpus_type == 'pc_tr':
            self.tags = (
                'Positive_regulation', 'Negative_regulation', 'Regulation', 'Binding', 'Pathway', 'Phosphorylation',
                'Gene_expression', 'Activation', 'Transport', 'Conversion', 'Localization', 'Inactivation',
                'Transcription',
                'Dissociation', 'Degradation', 'Ubiquitination', 'Acetylation', 'Dephosphorylation', 'Translation',
                'Methylation',
                'Demethylation', 'Deubiquitination', 'Hydroxylation', 'Deacetylation'
            )
            self.exclude_tags = (
                'Gene_or_gene_product', 'Simple_chemical', 'Complex', 'Cellular_component'
            )
            self.relations = (
                'Theme', 'Cause', 'Participant', 'Site', 'Product', 'ToLoc', 'AtLoc', 'FromLoc'
            )
        elif corpora.corpus_type == 'pc_en':
            self.tags = (
                'Positive_regulation', 'Negative_regulation', 'Regulation', 'Binding', 'Pathway', 'Phosphorylation',
                'Gene_expression', 'Activation', 'Transport', 'Conversion', 'Localization', 'Inactivation',
                'Transcription',
                'Dissociation', 'Degradation', 'Ubiquitination', 'Acetylation', 'Dephosphorylation', 'Translation',
                'Methylation',
                'Demethylation', 'Deubiquitination', 'Hydroxylation', 'Deacetylation',
                'Gene_or_gene_product', 'Simple_chemical', 'Complex', 'Cellular_component'
            )
            self.exclude_tags = (
                'Positive_regulation', 'Negative_regulation', 'Regulation', 'Binding', 'Pathway', 'Phosphorylation',
                'Gene_expression', 'Activation', 'Transport', 'Conversion', 'Localization', 'Inactivation',
                'Transcription',
                'Dissociation', 'Degradation', 'Ubiquitination', 'Acetylation', 'Dephosphorylation', 'Translation',
                'Methylation',
                'Demethylation', 'Deubiquitination', 'Hydroxylation', 'Deacetylation'
            )
            self.relations = (
                'Theme', 'Cause', 'Participant', 'Site', 'Product', 'ToLoc', 'AtLoc', 'FromLoc'
            )
        # -----------------------------------------EPI--------------------------------------------

        elif corpora.corpus_type == 'epi':
            self.tags = (
                'Methylation', 'Glycosylation', 'Acetylation', 'Ubiquitination', 'DNA_methylation', 'Catalysis',
                'Hydroxylation', 'Phosphorylation', 'Deacetylation', 'Deglycosylation', 'DNA_demethylation',
                'Deubiquitination', 'Demethylation', 'Dephosphorylation', 'Dehydroxylation',
                'Protein', 'Entity'
            )
            self.exclude_tags = (
            )
            self.relations = (
                'Theme', 'Site', 'Cause', 'Contextgene', 'Sidechain'
            )
        elif corpora.corpus_type == 'epi_tr':
            self.tags = (
                'Methylation', 'Glycosylation', 'Acetylation', 'Ubiquitination', 'DNA_methylation', 'Catalysis',
                'Hydroxylation', 'Phosphorylation', 'Deacetylation', 'Deglycosylation', 'DNA_demethylation',
                'Deubiquitination', 'Demethylation', 'Dephosphorylation', 'Dehydroxylation',
                'Protein', 'Entity'
            )
            self.exclude_tags = (
                'Protein', 'Entity'
            )
            self.relations = (
                'Theme', 'Site', 'Cause', 'Contextgene', 'Sidechain'
            )
        elif corpora.corpus_type == 'epi_en':
            self.tags = (
                'Methylation', 'Glycosylation', 'Acetylation', 'Ubiquitination', 'DNA_methylation', 'Catalysis',
                'Hydroxylation', 'Phosphorylation', 'Deacetylation', 'Deglycosylation', 'DNA_demethylation',
                'Deubiquitination', 'Demethylation', 'Dephosphorylation', 'Dehydroxylation',
                'Protein', 'Entity'
            )
            self.exclude_tags = (
                'Methylation', 'Glycosylation', 'Acetylation', 'Ubiquitination', 'DNA_methylation', 'Catalysis',
                'Hydroxylation', 'Phosphorylation', 'Deacetylation', 'Deglycosylation', 'DNA_demethylation',
                'Deubiquitination', 'Demethylation', 'Dephosphorylation', 'Dehydroxylation'
            )
            self.relations = (
                'Theme', 'Site', 'Cause', 'Contextgene', 'Sidechain'
            )

        # -----------------------------------------EZCAT--------------------------------------------
        elif corpora.corpus_type == "ezcat":
            self.tags = (
                "Activation",
                "BondFormation",
                "Cleavage",
                "ConformationalChange",
                "CouplingReaction",
                "Deprotonation",
                "Destabilisation",
                "ElectrophilicAttack",
                "HybridisationChange",
                "Inactivation",
                "Interaction",
                "Modulation",
                "NucleophilicAttack",
                "Others",
                "Protonation",
                "Release",
                "Stabilisation",
                "UncouplingReaction",
                "WholeReaction",
                "AminoAcid",
                "Cofactor",
                "EntityProperty",
                "Enzyme",
                "FunctionalGroup",
                "MethodCue",
                "NegationCue",
                "OtherCompound",
                "SpeculationCue",
            )
            self.exclude_tags = ()
            self.relations = (
                "Agent",
                "Cue",
                "EndPoint",
                "InitialPoint",
                "Means",
                "Theme",
            )
        elif corpora.corpus_type == "ezcat_tr":
            self.tags = (
                "Activation",
                "BondFormation",
                "Cleavage",
                "ConformationalChange",
                "CouplingReaction",
                "Deprotonation",
                "Destabilisation",
                "ElectrophilicAttack",
                "HybridisationChange",
                "Inactivation",
                "Interaction",
                "Modulation",
                "NucleophilicAttack",
                "Others",
                "Protonation",
                "Release",
                "Stabilisation",
                "UncouplingReaction",
                "WholeReaction",
                "AminoAcid",
                "Cofactor",
                "EntityProperty",
                "Enzyme",
                "FunctionalGroup",
                "MethodCue",
                "NegationCue",
                "OtherCompound",
                "SpeculationCue",
            )
            self.exclude_tags = (
                "AminoAcid",
                "Cofactor",
                "EntityProperty",
                "Enzyme",
                "FunctionalGroup",
                "MethodCue",
                "NegationCue",
                "OtherCompound",
                "SpeculationCue",
            )
            self.relations = (
                "Agent",
                "Cue",
                "EndPoint",
                "InitialPoint",
                "Means",
                "Theme",
            )
        elif corpora.corpus_type == "ezcat_en":
            self.tags = (
                "Activation",
                "BondFormation",
                "Cleavage",
                "ConformationalChange",
                "CouplingReaction",
                "Deprotonation",
                "Destabilisation",
                "ElectrophilicAttack",
                "HybridisationChange",
                "Inactivation",
                "Interaction",
                "Modulation",
                "NucleophilicAttack",
                "Others",
                "Protonation",
                "Release",
                "Stabilisation",
                "UncouplingReaction",
                "WholeReaction",
                "AminoAcid",
                "Cofactor",
                "EntityProperty",
                "Enzyme",
                "FunctionalGroup",
                "MethodCue",
                "NegationCue",
                "OtherCompound",
                "SpeculationCue",
            )
            self.exclude_tags = (
                "Activation",
                "BondFormation",
                "Cleavage",
                "ConformationalChange",
                "CouplingReaction",
                "Deprotonation",
                "Destabilisation",
                "ElectrophilicAttack",
                "HybridisationChange",
                "Inactivation",
                "Interaction",
                "Modulation",
                "NucleophilicAttack",
                "Others",
                "Protonation",
                "Release",
                "Stabilisation",
                "UncouplingReaction",
                "WholeReaction",
            )
            self.relations = (
                "Agent",
                "Cue",
                "EndPoint",
                "InitialPoint",
                "Means",
                "Theme",
            )

        # -----------------------------------------LCGENES--------------------------------------------

        elif corpora.corpus_type == 'lcgenes':
            self.tags = (
                'MMLite', 'GGPs', 'Disorder', 'Subject', 'Pharmacological_substance',
                'Cell', 'Method_cue', 'Anatomical_entity', 'Cell_component', 'Organic_compound_other',
                'Inorganic_compound',
                'Artificial_process', 'Molecular_function', 'Biological_process', 'Cellular_process', 'Regulation',
            )
            self.exclude_tags = (
            )
            self.relations = (
            )
        elif corpora.corpus_type == 'lcgenes_tr':
            self.tags = (
                'MMLite', 'GGPs', 'Disorder', 'Subject', 'Pharmacological_substance',
                'Cell', 'Method_cue', 'Anatomical_entity', 'Cell_component', 'Organic_compound_other',
                'Inorganic_compound',
                'Artificial_process', 'Molecular_function', 'Biological_process', 'Cellular_process', 'Regulation',
            )
            self.exclude_tags = (
                'MMLite', 'GGPs', 'Disorder', 'Subject', 'Pharmacological_substance',
                'Cell', 'Method_cue', 'Anatomical_entity', 'Cell_component', 'Organic_compound_other',
                'Inorganic_compound',
            )
            self.relations = (
            )
        elif corpora.corpus_type == 'lcgenes_en':
            self.tags = (
                'MMLite', 'GGPs', 'Disorder', 'Subject', 'Pharmacological_substance',
                'Cell', 'Method_cue', 'Anatomical_entity', 'Cell_component', 'Organic_compound_other',
                'Inorganic_compound',
                'Artificial_process', 'Molecular_function', 'Biological_process', 'Cellular_process', 'Regulation',
            )
            self.exclude_tags = (
                'Artificial_process', 'Molecular_function', 'Biological_process', 'Cellular_process', 'Regulation',
            )
            self.relations = (
            )
        # -----------------------------------------IPF--------------------------------------------
        elif corpora.corpus_type == 'ipf':
            self.tags = (
                'MMLite', 'GGPs', 'Disorder', 'Subject', 'Anatomical_entity', 'Cell', 'Method_cue',
                'Pharmacological_substance', 'Entity_Property', 'Organic_compound_other', 'Cell_component',
                'Inorganic_compound',
                'Artificial_process', 'Positive_regulation', 'Gene_expression', 'Negative_regulation',
                'Cellular_process', 'Biological_process', 'Pathway', 'Molecular_function', 'Regulation', 'Migration',
                'Localization',
            )
            self.exclude_tags = (
            )
            self.relations = (
                'Theme', 'Participant', 'Cause', 'disorder', 'atLoc',
            )
        elif corpora.corpus_type == 'ipf_tr':
            self.tags = (
                'MMLite', 'GGPs', 'Disorder', 'Subject', 'Anatomical_entity', 'Cell', 'Method_cue',
                'Pharmacological_substance', 'Entity_Property', 'Organic_compound_other', 'Cell_component',
                'Inorganic_compound',
                'Artificial_process', 'Positive_regulation', 'Gene_expression', 'Negative_regulation',
                'Cellular_process', 'Biological_process', 'Pathway', 'Molecular_function', 'Regulation', 'Migration',
                'Localization',
            )
            self.exclude_tags = (
                'MMLite', 'GGPs', 'Disorder', 'Subject', 'Anatomical_entity', 'Cell', 'Method_cue',
                'Pharmacological_substance', 'Entity_Property', 'Organic_compound_other', 'Cell_component',
                'Inorganic_compound',
            )
            self.relations = (
                'Theme', 'Participant', 'Cause', 'disorder', 'atLoc',
            )
        elif corpora.corpus_type == 'ipf_en':
            self.tags = (
                'MMLite', 'GGPs', 'Disorder', 'Subject', 'Anatomical_entity', 'Cell', 'Method_cue',
                'Pharmacological_substance', 'Entity_Property', 'Organic_compound_other', 'Cell_component',
                'Inorganic_compound',
                'Artificial_process', 'Positive_regulation', 'Gene_expression', 'Negative_regulation',
                'Cellular_process', 'Biological_process', 'Pathway', 'Molecular_function', 'Regulation', 'Migration',
                'Localization',
            )
            self.exclude_tags = (
                'Artificial_process', 'Positive_regulation', 'Gene_expression', 'Negative_regulation',
                'Cellular_process', 'Biological_process', 'Pathway', 'Molecular_function', 'Regulation', 'Migration',
                'Localization',
            )
            self.relations = (
                'Theme', 'Participant', 'Cause', 'disorder', 'atLoc',
            )

        # -----------------------------------------GPCR--------------------------------------------
        elif corpora.corpus_type == 'gpcr':
            self.tags = (
                'Protein', 'GPCR', 'GPCR-ligand', 'Chemical', 'Cell', 'G-protein', 'Disease',
                'Cell-component', 'Organism', 'Anatomy', 'Entity',
                'Regulation', 'Positive_regulation', 'Biological_process', 'Negative_regulation', 'Pathway', 'Binding',
                'Gene_expression', 'Artificial_process', 'Localization', 'Phosphorylation', 'Internalization',
                'Biosynthesis', 'Conformational-change', 'Degradation', 'Conversion', 'Transportation', 'Dissociation',
                'Transcription', 'Dephosphorylation', 'Translation',
            )
            self.exclude_tags = (
            )
            self.relations = (
                'Theme', 'Cause', 'Participant', 'AtLoc', 'Site', 'Product', 'ToLoc', 'FromLoc',
            )
        elif corpora.corpus_type == 'gpcr_tr':
            self.tags = (
                'Protein', 'GPCR', 'GPCR-ligand', 'Chemical', 'Cell', 'G-protein', 'Disease',
                'Cell-component', 'Organism', 'Anatomy', 'Entity',
                'Regulation', 'Positive_regulation', 'Biological_process', 'Negative_regulation', 'Pathway', 'Binding',
                'Gene_expression', 'Artificial_process', 'Localization', 'Phosphorylation', 'Internalization',
                'Biosynthesis', 'Conformational-change', 'Degradation', 'Conversion', 'Transportation', 'Dissociation',
                'Transcription', 'Dephosphorylation', 'Translation',
            )
            self.exclude_tags = (
                'Protein', 'GPCR', 'GPCR-ligand', 'Chemical', 'Cell', 'G-protein', 'Disease',
                'Cell-component', 'Organism', 'Anatomy', 'Entity',
            )
            self.relations = (
                'Theme', 'Cause', 'Participant', 'AtLoc', 'Site', 'Product', 'ToLoc', 'FromLoc',
            )
        elif corpora.corpus_type == 'gpcr_en':
            self.tags = (
                'Protein', 'GPCR', 'GPCR-ligand', 'Chemical', 'Cell', 'G-protein', 'Disease',
                'Cell-component', 'Organism', 'Anatomy', 'Entity',
                'Regulation', 'Positive_regulation', 'Biological_process', 'Negative_regulation', 'Pathway', 'Binding',
                'Gene_expression', 'Artificial_process', 'Localization', 'Phosphorylation', 'Internalization',
                'Biosynthesis', 'Conformational-change', 'Degradation', 'Conversion', 'Transportation', 'Dissociation',
                'Transcription', 'Dephosphorylation', 'Translation',
            )
            self.exclude_tags = (
                'Regulation', 'Positive_regulation', 'Biological_process', 'Negative_regulation', 'Pathway', 'Binding',
                'Gene_expression', 'Artificial_process', 'Localization', 'Phosphorylation', 'Internalization',
                'Biosynthesis', 'Conformational-change', 'Degradation', 'Conversion', 'Transportation', 'Dissociation',
                'Transcription', 'Dephosphorylation', 'Translation',
            )
            self.relations = (
                'Theme', 'Cause', 'Participant', 'AtLoc', 'Site', 'Product', 'ToLoc', 'FromLoc',
            )

        # -----------------------------------------GE04--------------------------------------------
        elif corpora.corpus_type == 'genia04':
            self.tags = (
                "protein",
                "DNA",
                "cell_type",
                "cell_line",
                "RNA",
            )
            self.exclude_tags = (
            )
            self.relations = (
            )

        self.actual_tags = (tag for tag in self.tags if tag not in self.exclude_tags)  # Not use set to keep order
        for g, s in corpora.docs:
            evaluator = SingleEvaluator(g, s, 2, mode, tag_type, verbose=verbose, exclude_tags=self.exclude_tags)
            for target in ('tags', 'relations'):
                for score in ('tp', 'fp', 'fn'):
                    self.scores[target][score] += evaluator.scores[target][score]
                measures = Measures(tp=evaluator.scores[target]['tp'],
                                    fp=evaluator.scores[target]['fp'],
                                    fn=evaluator.scores[target]['fn'],
                                    tn=evaluator.scores[target]['tn'])
                for score in ('precision', 'recall', 'f1'):
                    fn = getattr(measures, score)
                    self.scores[target]['macro'][score] += fn()

        for target in ('tags', 'relations'):
            # Normalization
            for key in self.scores[target]['macro'].keys():
                self.scores[target]['macro'][key] = \
                    self.scores[target]['macro'][key] / len(corpora.docs)

            measures = Measures(tp=self.scores[target]['tp'],
                                fp=self.scores[target]['fp'],
                                fn=self.scores[target]['fn'],
                                tn=self.scores[target]['tn'])
            for key in self.scores[target]['micro'].keys():
                fn = getattr(measures, key)
                self.scores[target]['micro'][key] = fn()


def evaluate(corpora, mode='strict', verbose=False):
    """Run the evaluation by considering only files in the two folders."""
    assert mode in ('strict', 'lenient')
    evaluator_s = MultipleEvaluator(corpora, verbose)
    if corpora.track == 1:
        macro_f1, macro_auc = 0, 0
        print('{:*^96}'.format(' TRACK 1 '))
        print('{:20}  {:-^30}    {:-^22}    {:-^14}'.format('', ' met ',
                                                            ' not met ',
                                                            ' overall '))
        print('{:20}  {:6}  {:6}  {:6}  {:6}    {:6}  {:6}  {:6}    {:6}  {:6}'.format(
            '', 'Prec.', 'Rec.', 'Speci.', 'F(b=1)', 'Prec.', 'Rec.', 'F(b=1)', 'F(b=1)', 'AUC'))
        for tag in evaluator_s.tags:
            print(
                '{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}'.format(
                    tag.capitalize(),
                    evaluator_s.scores[(tag, 'met', 'p')],
                    evaluator_s.scores[(tag, 'met', 'r')],
                    evaluator_s.scores[(tag, 'met', 'specificity')],
                    evaluator_s.scores[(tag, 'met', 'f1')],
                    evaluator_s.scores[(tag, 'not met', 'p')],
                    evaluator_s.scores[(tag, 'not met', 'r')],
                    evaluator_s.scores[(tag, 'not met', 'f1')],
                    (evaluator_s.scores[(tag, 'met', 'f1')] + evaluator_s.scores[(tag, 'not met', 'f1')]) / 2,
                    evaluator_s.scores[(tag, 'met', 'auc')]))
            macro_f1 += (evaluator_s.scores[(tag, 'met', 'f1')] + evaluator_s.scores[(tag, 'not met', 'f1')]) / 2
            macro_auc += evaluator_s.scores[(tag, 'met', 'auc')]
        print('{:20}  {:-^30}    {:-^22}    {:-^14}'.format('', '', '', ''))
        m = Measures(tp=evaluator_s.values['met']['tp'],
                     fp=evaluator_s.values['met']['fp'],
                     fn=evaluator_s.values['met']['fn'],
                     tn=evaluator_s.values['met']['tn'])
        nm = Measures(tp=evaluator_s.values['not met']['tp'],
                      fp=evaluator_s.values['not met']['fp'],
                      fn=evaluator_s.values['not met']['fn'],
                      tn=evaluator_s.values['not met']['tn'])
        print(
            '{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}'.format(
                'Overall (micro)', m.precision(), m.recall(), m.specificity(),
                m.f1(), nm.precision(), nm.recall(), nm.f1(),
                (m.f1() + nm.f1()) / 2, m.auc()))
        print(
            '{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}'.format(
                'Overall (macro)',
                evaluator_s.scores[('macro', 'met', 'p')],
                evaluator_s.scores[('macro', 'met', 'r')],
                evaluator_s.scores[('macro', 'met', 'specificity')],
                evaluator_s.scores[('macro', 'met', 'f1')],
                evaluator_s.scores[('macro', 'not met', 'p')],
                evaluator_s.scores[('macro', 'not met', 'r')],
                evaluator_s.scores[('macro', 'not met', 'f1')],
                macro_f1 / len(evaluator_s.tags),
                evaluator_s.scores[('macro', 'met', 'auc')]))
        print()
        print('{:>20}  {:^74}'.format('', '  {} files found  '.format(len(corpora.docs))))
    else:
        evaluator_l = MultipleEvaluator(corpora, mode='lenient', verbose=verbose)
        print('{:*^70}'.format(' TRACK 2 '))
        print('{:20}  {:-^22}    {:-^22}'.format('', ' strict ', ' lenient '))
        print('{:20}  {:6}  {:6}  {:6}    {:6}  {:6}  {:6}'.format('', 'Prec.',
                                                                   'Rec.',
                                                                   'F(b=1)',
                                                                   'Prec.',
                                                                   'Rec.',
                                                                   'F(b=1)'))
        for tag in evaluator_s.actual_tags:
            evaluator_tag_s = MultipleEvaluator(corpora, tag, verbose=verbose)
            evaluator_tag_l = MultipleEvaluator(corpora, tag, mode='lenient', verbose=verbose)
            print(
                '{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}   {:>5}  {:>5}   {:>5}   {:>5}  {:>5}   {:>5}'.format(
                    tag.capitalize(),
                    evaluator_tag_s.scores['tags']['micro']['precision'],
                    evaluator_tag_s.scores['tags']['micro']['recall'],
                    evaluator_tag_s.scores['tags']['micro']['f1'],
                    evaluator_tag_l.scores['tags']['micro']['precision'],
                    evaluator_tag_l.scores['tags']['micro']['recall'],
                    evaluator_tag_l.scores['tags']['micro']['f1'],
                    evaluator_tag_s.scores['tags']['tp'] +
                    evaluator_tag_s.scores['tags']['fp'],
                    evaluator_tag_s.scores['tags']['tp'] +
                    evaluator_tag_s.scores['tags']['fn'],
                    evaluator_tag_s.scores['tags']['tp'],
                    evaluator_tag_l.scores['tags']['tp'] +
                    evaluator_tag_l.scores['tags']['fp'],
                    evaluator_tag_l.scores['tags']['tp'] +
                    evaluator_tag_l.scores['tags']['fn'],
                    evaluator_tag_l.scores['tags']['tp']))
        print('{:>20}  {:-^48}'.format('', ''))
        print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}'.format(
            'Overall (micro)',
            evaluator_s.scores['tags']['micro']['precision'],
            evaluator_s.scores['tags']['micro']['recall'],
            evaluator_s.scores['tags']['micro']['f1'],
            evaluator_l.scores['tags']['micro']['precision'],
            evaluator_l.scores['tags']['micro']['recall'],
            evaluator_l.scores['tags']['micro']['f1']))
        print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}'.format(
            'Overall (macro)',
            evaluator_s.scores['tags']['macro']['precision'],
            evaluator_s.scores['tags']['macro']['recall'],
            evaluator_s.scores['tags']['macro']['f1'],
            evaluator_l.scores['tags']['macro']['precision'],
            evaluator_l.scores['tags']['macro']['recall'],
            evaluator_l.scores['tags']['macro']['f1']))
        print()

        print('{:*^70}'.format(' RELATIONS '))
        for rel in evaluator_s.relations:
            evaluator_tag_s = MultipleEvaluator(corpora, rel, mode='strict', verbose=verbose)
            evaluator_tag_l = MultipleEvaluator(corpora, rel, mode='lenient', verbose=verbose)
            print(
                '{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}    {:>5}  {:>5}   {:>5}   {:>5}  {:>5}   {:>5}'.format(
                    '{}'.format(rel),
                    evaluator_tag_s.scores['relations']['micro']['precision'],
                    evaluator_tag_s.scores['relations']['micro']['recall'],
                    evaluator_tag_s.scores['relations']['micro']['f1'],
                    evaluator_tag_l.scores['relations']['micro']['precision'],
                    evaluator_tag_l.scores['relations']['micro']['recall'],
                    evaluator_tag_l.scores['relations']['micro']['f1'],
                    evaluator_tag_s.scores['relations']['tp'] +
                    evaluator_tag_s.scores['relations']['fp'],
                    evaluator_tag_s.scores['relations']['tp'] +
                    evaluator_tag_s.scores['relations']['fn'],
                    evaluator_tag_s.scores['relations']['tp'],
                    evaluator_tag_l.scores['relations']['tp'] +
                    evaluator_tag_l.scores['relations']['fp'],
                    evaluator_tag_l.scores['relations']['tp'] +
                    evaluator_tag_l.scores['relations']['fn'],
                    evaluator_tag_l.scores['relations']['tp']))
        print('{:>20}  {:-^48}'.format('', ''))
        print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}'.format(
            'Overall (micro)',
            evaluator_s.scores['relations']['micro']['precision'],
            evaluator_s.scores['relations']['micro']['recall'],
            evaluator_s.scores['relations']['micro']['f1'],
            evaluator_l.scores['relations']['micro']['precision'],
            evaluator_l.scores['relations']['micro']['recall'],
            evaluator_l.scores['relations']['micro']['f1']))
        print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}'.format(
            'Overall (macro)',
            evaluator_s.scores['relations']['macro']['precision'],
            evaluator_s.scores['relations']['macro']['recall'],
            evaluator_s.scores['relations']['macro']['f1'],
            evaluator_l.scores['relations']['macro']['precision'],
            evaluator_l.scores['relations']['macro']['recall'],
            evaluator_l.scores['relations']['macro']['f1']))
        print()
        print('{:20}{:^48}'.format('', '  {} files found  '.format(len(corpora.docs))))


class Corpora(object):

    def __init__(self, corpus_type, folder1, folder2, track_num):
        extensions = {1: '*.xml', 2: '*.ann'}
        file_ext = extensions[track_num]
        self.track = track_num
        self.folder1 = folder1
        self.folder2 = folder2
        self.corpus_type = corpus_type
        files1 = set([os.path.basename(f) for f in glob.glob(
            os.path.join(folder1, file_ext))])
        # print(files1)
        files2 = set([os.path.basename(f) for f in glob.glob(
            os.path.join(folder2, file_ext))])
        # print(files2)
        common_files = files1 & files2  # intersection
        if not common_files:
            print('ERROR: None of the files match.')
        else:
            if files1 - common_files:
                print('Files skipped in {}:'.format(self.folder1))
                print(', '.join(sorted(list(files1 - common_files))))
            if files2 - common_files:
                print('Files skipped in {}:'.format(self.folder2))
                print(', '.join(sorted(list(files2 - common_files))))
        self.docs = []
        for file in common_files:
            if track_num == 1:
                g = RecordTrack1(os.path.join(self.folder1, file))
                s = RecordTrack1(os.path.join(self.folder2, file))
            else:
                g = RecordTrack2(os.path.join(self.folder1, file))
                s = RecordTrack2(os.path.join(self.folder2, file))
            self.docs.append((g, s))


def main(corpus_type, f1, f2, track, verbose):
    """Where the magic begins."""
    corpora = Corpora(corpus_type, f1, f2, track)
    if corpora.docs:
        evaluate(corpora, verbose=verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='n2c2: Evaluation script for Track 2')
    parser.add_argument('folder1', help='First data folder path (gold)')
    parser.add_argument('folder2', help='Second data folder path (system)')
    parser.add_argument('--ner-eval-corpus', dest='corpus_type', type=str, required=True,
                        help='ace / cg / cg_tr / ace_tr')
    args = parser.parse_args()
    main(args.corpus_type, os.path.abspath(args.folder1), os.path.abspath(args.folder2), 2, False)
