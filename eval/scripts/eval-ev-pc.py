#!/usr/bin/env python

# Implementation of BioNLP Shared Task evaluation.

# Copyright (c) 2010-2013 BioNLP Shared Task organizers
# This script is distributed under the open source MIT license:
# http://www.opensource.org/licenses/mit-license

import sys
import re
import os
import optparse

# allowed types for entities, events, etc. (task-specific settings)
given_types = set([
        "Simple_chemical",
        "Gene_or_gene_product",
        "Complex",
        "Cellular_component",
        ])

entity_types = set([
        "Simple_chemical",
        "Gene_or_gene_product",
        "Complex",
        "Cellular_component",
        ])

event_types  = set([
        "Conversion",
        "Phosphorylation",
        "Dephosphorylation",
        "Acetylation",
        "Deacetylation",
        "Methylation",
        "Demethylation",
        "Ubiquitination",
        "Deubiquitination",
        "Localization",
        "Transport",
        "Gene_expression",
        "Transcription",
        "Translation",
        "Degradation",
        "Activation",
        "Inactivation",
        "Binding",
        "Dissociation",
        "Regulation", 
        "Positive_regulation",
        "Negative_regulation",
        "Pathway",
        ])

valid_argument_types = {
}

output_event_type_order = [
    "Conversion",
    "Phosphorylation",
    "Dephosphorylation",
    "Acetylation",
    "Deacetylation",
    "Methylation",
    "Demethylation",
    "Ubiquitination",
    "Deubiquitination",
    "Localization",
    "Transport",
    "Gene_expression",
    "Transcription",
    "Translation",
    ' =[SIMPLE-TOTAL]= ',
    "Degradation",
    "Activation",
    "Inactivation",
    "Binding",
    "Dissociation",
    "Pathway",
    '==[NONREG-TOTAL]==',
    "Regulation", 
    "Positive_regulation",
    "Negative_regulation",
    ' ==[REG-TOTAL]==  ',
    ' ====[TOTAL]====  ',
    ]

subtotal_event_set = {
    ' =[SIMPLE-TOTAL]= ': [
            "Conversion",
            "Phosphorylation",
            "Dephosphorylation",
            "Acetylation",
            "Deacetylation",
            "Methylation",
            "Demethylation",
            "Ubiquitination",
            "Deubiquitination",
            "Localization",
            "Transport",
            "Gene_expression",
            "Transcription",
            "Translation",
            ],
    '==[NONREG-TOTAL]==' : [
            "Conversion",
            "Phosphorylation",
            "Dephosphorylation",
            "Acetylation",
            "Deacetylation",
            "Methylation",
            "Demethylation",
            "Ubiquitination",
            "Deubiquitination",
            "Localization",
            "Transport",
            "Gene_expression",
            "Transcription",
            "Translation",
            "Degradation",
            "Activation",
            "Inactivation",
            "Binding",
            "Dissociation",
            "Pathway",
            ],
    ' ==[REG-TOTAL]==  ' : [
            'Regulation',
            'Positive_regulation',
            'Negative_regulation',
            ],
    ' ====[TOTAL]====  ' : [
            "Conversion",
            "Phosphorylation",
            "Dephosphorylation",
            "Acetylation",
            "Deacetylation",
            "Methylation",
            "Demethylation",
            "Ubiquitination",
            "Deubiquitination",
            "Localization",
            "Transport",
            "Gene_expression",
            "Transcription",
            "Translation",
            "Degradation",
            "Activation",
            "Inactivation",
            "Binding",
            "Dissociation",
            "Pathway",
            "Regulation", 
            "Positive_regulation",
            "Negative_regulation",
            ]    
}

# allowed types for text-bound annotation
textbound_types = entity_types | event_types

# allowed types for modification annotation
modification_types = set(["Negation", "Speculation"])

# punctuation characters halting "soft match" span extension
punctuation_chars = set(".!?,\"'")


# generator, returns all permutations of the given list.
# Following http://code.activestate.com/recipes/252178/.
def all_permutations(lst):
    if len(lst) <= 1:
        yield lst
    else:
        for perm in all_permutations(lst[1:]):
            for i in range(len(perm) + 1):
                yield perm[:i] + lst[0:1] + perm[i:]


# returns (prec, rec, F) 3-tuple given TPa, TPg, FP, FN.
# (See comment after "scoring starts here" for details.)
def prec_rec_F(match_answer, match_gold, false_positive, false_negative):
    precision, recall, F = 0, 0, 0
    if match_answer + false_positive > 0:
        precision = float(match_answer) / (match_answer + false_positive)
        # print("Precision:", 100.0 * precision, match_answer, false_positive)
    if match_gold + false_negative > 0:
        recall = float(match_gold) / (match_gold + false_negative)
        # print("Recall:", 100.0 * recall, match_gold, false_negative)
    if precision + recall > 0:
        F = 2 * precision * recall / (precision + recall)
        # print("F:", 100.0 * F)

    # corner case: if gold is empty, an empty answer gives a perfect
    # score
    if match_answer + match_gold + false_positive + false_negative == 0:
        precision = recall = F = 1

    # return percentages
    return 100.0 * precision, 100.0 * recall, 100.0 * F


# allowed arguments
def valid_argument_name(a):
    return re.match(r'^Cause|CSite|ToLoc|AtLoc|FromLoc|(Theme|Site|Product|Participant)\d*$', a) is not None

# allowed (argument, referred-type) combinations
def valid_argument_type(arg, reftype):
    if arg == "Cause" or re.match(r'^(Theme|Product)\d*$', arg):
        return reftype in given_types or reftype in event_types
    elif arg in ("ToLoc", "AtLoc", "FromLoc"):
        return reftype in ("Cellular_component", )
    elif re.match(r'^C?Site\d*$', arg):
        return reftype in ("Simple_chemical", )
    elif re.match(r'^Participant\d*$', arg):
        return reftype in given_types
    else:
        assert False, "INTERNAL ERROR: unexpected argument type %s" % arg


# represents a text-bound annotation (entity/event trigger)
class Textbound:
    def __init__(self, id, type, start, end, text):
        self.id, self.type, self.start, self.end, self.text = id, type, start, end, text
        self.is_trigger = self.type in event_types
        self.equivs = None
        self.extended_start, self.extended_end = None, None
        # for merged output, store matching
        self.matching = []

    def verify_text(self, document_text, fn):
        # checks that text matches the span in the given document text
        # identified by (start, end].
        assert self.start < len(document_text) and self.end <= len(
            document_text), "FORMAT ERROR: textbound offsets extend over document length: '%s' in %s" % (
        self.to_string(), fn)
        assert document_text[self.start:self.end].strip().replace("\n",
                                                                  " ") == self.text.strip(), "FORMAT ERROR: text '%s' referenced by [start, end) doesn't match given text '%s': '%s' in %s" % (
        document_text[self.start:self.end], self.text, self.to_string(), fn)

    def identify_extended_span(self, document_text, reserved_index):
        # find "extended" start and end points. reserved_index
        # values are taked to be "off limits" and stop extension.
        estart, eend = self.start, self.end

        # implementation copying evaluation.pl

        # try to back off start by one, skipping possible intervening space
        for i in range(0, 1):
            estart -= 1
            # avoid invalid and reserved
            if estart < 0 or estart in reserved_index:
                estart += 1
        # back off until space or punctuation
        while (estart - 1 >= 0 and not estart - 1 in reserved_index and
               not document_text[estart - 1].isspace() and not document_text[estart - 1] in punctuation_chars):
            estart -= 1

        # symmetric for the end
        for i in range(0, 1):
            eend += 1
            if eend > len(document_text) or eend - 1 in reserved_index:
                eend -= 1
        while (eend < len(document_text) and not eend in reserved_index and
               not document_text[eend].isspace() and not document_text[eend] in punctuation_chars):
            eend += 1

        self.extended_start, self.extended_end = estart, eend

        # sanity: the extended range must be at least as large as the original
        assert self.extended_start <= self.start, "INTERNAL ERROR"
        assert self.extended_end >= self.end, "INTERNAL ERROR"

    def resolve_idrefs(self, annotation_by_id):
        # none to resolve
        pass

    def to_string(self):
        return "%s\t%s %d %d\t%s" % (self.id, self.type, self.start, self.end, self.text)

    def to_notbid_string(self):
        # the type is too obvious in my current work
        # return '%s:"%s"' % (self.type, self.text)
        return '"%s"[%d-%d]' % (self.text, self.start, self.end)

    def matches_self(self, t):
        # match, doesn't consider 'Equiv' textbounds but only self.
        global options

        if self.type != t.type:
            return False

        if options.softboundary and self.is_trigger:
            # containment in extended span.
            # As we're now allowing matches() to be invoked also on
            # given entities, we need to check which one is extended
            # and check against that.
            #             assert t.extended_start is not None and t.extended_end is not None, "INTERNAL ERROR"
            #             return t.extended_start <= self.start and t.extended_end >= self.end
            if t.extended_start is not None:
                assert t.extended_end is not None, "INTERNAL ERROR"
                return t.extended_start <= self.start and t.extended_end >= self.end
            else:
                assert self.extended_start is not None and self.extended_end is not None, "INTERNAL ERROR"
                return self.extended_start <= t.start and self.extended_end >= t.end
        else:
            # strict equality
            return self.start == t.start and self.end == t.end

    def matches_impl(self, t):
        # Textbound can only match Textbound
        if not isinstance(t, Textbound):
            return False

        # a match between any 'Equiv' textbounds is a match
        self_equiv_entities = [self]
        other_equiv_entities = [t]
        if self.equivs is not None:
            self_equiv_entities += self.equivs
        if t.equivs is not None:
            other_equiv_entities += t.equivs

        for se in self_equiv_entities:
            for te in other_equiv_entities:
                if se.matches_self(te):
                    return True

        return False

    # dummy to allow uniform argument structure with Event matches().
    def matches(self, t, dummy=False):
        m = self.matches_impl(t)
        if m and t is not self and t not in self.matching:
            self.matching.append(t)
        return m


# represents Equiv annotation (equivalent entities)
class Equiv:
    def __init__(self, ids):
        self.ids = ids
        self.entities = None

    def resolve_idrefs(self, annotation_by_id):
        self.entities = []
        for id in self.ids:
            assert id in annotation_by_id, "ERROR: undefined ID %s referenced in Equiv" % id
            e = annotation_by_id[id]

            # this constraint now relaxed.
            #             # check: only given types can be Equiv
            #             assert e.type in given_types, "ERROR: Equiv for non-given, type %s ID %s" % (e.type, id)
            assert e not in self.entities, "ERROR: %s with ID %s occurs multiple times in Equiv" % (e.type, id)
            self.entities.append(e)

    def to_string(self):
        return "*\tEquiv %s" % " ".join(self.ids)


# represents event annotation
class Event:
    def __init__(self, id, type, tid, args):
        self.id, self.type, self.tid, self.args = id, type, tid, args
        self.trigger = None
        self.matching = []

        # sanity: no argument can refer back to the event itself
        for arg, aid in self.args:
            assert aid != self.id, "ERROR: event ID %s contains argument %s referring to itself" % (self.id, arg)

    def resolve_idrefs(self, annotation_by_id):
        global options

        assert self.tid in annotation_by_id, "ERROR: undefined trigger ID %s referenced in event %s" % (
        self.tid, self.id)
        self.trigger = annotation_by_id[self.tid]
        assert self.trigger.type == self.type, "ERROR: trigger %s type disagrees with event %s type" % (
        self.tid, self.id)

        # Themes and Sites will be stored as a list of (argnum, Theme,
        # Site) tuples. Construct theme_site using a dict indexed by
        # arg name containing a list of the values: multiple Themes
        # may share the same number as long as there are no Sites (the
        # connection between Theme and Site must be unambiguous).
        # Other optionally repeated args, Participant and Instrument,
        # are also stored as lists.
        self.arg = {}
        self.theme_sites = []
        self.participants = []
        self.products = []
        theme_site_dict = {}
        theme_site_nums = set()

        for arg, aid in self.args:
            if aid not in annotation_by_id:
                assert aid in annotation_by_id, "ERROR: undefined ID %s referenced in event %s" % (aid, self.id)

            ref = annotation_by_id[aid]
            if not valid_argument_type(arg, ref.type):
                assert valid_argument_type(arg, ref.type), "ERROR: argument %s for %s event %s has invalid type %s" % (
                arg, self.type, self.id, ref.type)

            m = re.match(r'^(Theme|Site)(\d*)$', arg)
            m2 = re.match(r'^(Participant)(\d*)$', arg)
            m3 = re.match(r'^(Product)(\d*)$', arg)
            if m:
                argname, argnum = m.groups()
                if arg not in theme_site_dict:
                    theme_site_dict[arg] = []
                theme_site_dict[arg].append(ref)
                theme_site_nums.add(argnum)
            elif m2:
                argname, argnum = m2.groups()
                self.participants.append((argnum, ref))
            elif m3:
                argname, argnum = m3.groups()
                self.products.append((argnum, ref))
            else:
                assert arg not in self.arg, "ERROR: event %s has multiple %s arguments" % (self.id, arg)
                self.arg[arg] = ref

        # construct theme_sites list, with sanity checks
        for argnum in theme_site_nums:
            t_arg, s_arg = "Theme%s" % argnum, "Site%s" % argnum

            # there must be a Theme for each Theme/Site number that appears,
            # except in split event eval and for Mutation events.
            if not options.spliteventeval and self.type != 'Mutation':
                assert t_arg in theme_site_dict, "ERROR: event %s has Site%s without Theme%s" % (
                self.id, argnum, argnum)

            # Theme-Site pairings must be unambiguous
            s_val = None
            if s_arg in theme_site_dict:
                assert len(theme_site_dict[
                               s_arg]) == 1, "ERROR: event %s has multiple Site%s arguments, Site-Theme pairing is ambiguous" % (
                self.id, argnum)
                # ("not in" for spliteventeval)
                assert t_arg not in theme_site_dict or len(theme_site_dict[
                                                               t_arg]) == 1, "ERROR: event %s has Site%s and multiple Theme%s arguments, Site-Theme pairing is ambiguous" % (
                self.id, argnum, argnum)
                s_val = theme_site_dict[s_arg][0]

            if t_arg in theme_site_dict:
                for t_val in theme_site_dict[t_arg]:
                    self.theme_sites.append((argnum, t_val, s_val))
            else:
                assert options.spliteventeval or self.type == 'Mutation', "INTERNAL ERROR"
                assert s_val is not None, "INTERNAL ERROR: neither theme nor site"
                self.theme_sites.append((argnum, None, s_val))

        # finally, there can be no CSite without a corresponding Cause.
        # (except in split event eval mode)
        # if not options.spliteventeval:
        #     assert not ("CSite" in self.arg and not "Cause" in self.arg), "ERROR: event %s has CSite without Cause" % self.id

    def clear_matching(self):
        self.matching = []
        self.partially_matched = []
        self.partially_matched_by = []

    def has_matched(self):
        return len(self.matching) != 0

    def has_partially_matched(self):
        return len(self.partially_matched) != 0

    def mark_matching(self, matching):
        assert matching is not None, "INTERNAL ERROR: None given as matched event"
        self.matching.append(matching)

    def mark_partially_matched(self, matching):
        assert matching is not None, "INTERNAL ERROR: None given as matched event"
        self.partially_matched.append(matching)

    def mark_partially_matched_by(self, matching):
        assert matching is not None, "INTERNAL ERROR: None given as matched event"
        self.partially_matched_by.append(matching)

    def to_string(self):
        return "%s\t%s:%s %s" % (self.id, self.type, self.trigger.id, " ".join(["%s:%s" % a for a in self.args]))

    def to_notbid_string(self):
        # returns a string with no IDs for textbounds
        argstrs = []
        for ts in self.theme_sites:
            argstrs.append("%s:%s" % ("Theme", ts[1].to_notbid_string()))
            if ts[2] is not None:
                argstrs.append("%s:%s" % ("Site", ts[2].to_notbid_string()))
        for num, ref in self.participants:
            argstrs.append("%s:%s" % ("Participant" + num, ref.to_notbid_string()))
        for num, ref in self.products:
            argstrs.append("%s:%s" % ("Product" + num, ref.to_notbid_string()))
        for a in self.arg:
            argstrs.append("%s:%s" % (a, self.arg[a].to_notbid_string()))

        return '%s:%s\t(%s)' % (self.type, self.trigger.to_notbid_string(), "\t".join(argstrs))

    def matches(self, e, match_theme_only=False, match_partial=False):
        global options

        # Event can only match Event
        if not isinstance(e, Event):
            return False

        # events with different types cannot match
        if self.type != e.type:
            return False

        # triggers must match
        if not self.trigger.matches(e.trigger):
            return False

        # determine fixed argument matches
        self_matched_args, other_matched_args = [], []
        self_only_args, other_only_args = [], []

        for a in self.arg:
            if a in e.arg:
                if self.arg[a].matches(e.arg[a], options.partialrecursive):
                    self_matched_args.append(a)
                    other_matched_args.append(a)
                else:
                    # both have the argument, but different values
                    self_only_args.append(a)
                    other_only_args.append(a)
            else:
                self_only_args.append(a)
        for a in e.arg:
            if a in self.arg:
                # handled above
                pass
            else:
                other_only_args.append(a)

        # participant matches
        matched_self_p, matched_other_p = [], []
        for sp in self.participants:
            for op in e.participants:
                if op in matched_other_p:
                    continue  # only match once
                if sp[1].matches(op[1], options.partialrecursive):
                    matched_self_p.append(sp)
                    matched_other_p.append(op)
                    break
        for sp in [p for p in self.participants if p not in matched_self_p]:
            self_only_args.append('Participant' + sp[0])
        for op in [p for p in e.participants if p not in matched_other_p]:
            other_only_args.append('Participant' + op[0])

        # product matches
        matched_self_i, matched_other_i = [], []
        for sp in self.products:
            for op in e.products:
                if op in matched_other_i:
                    continue  # only match once
                if sp[1].matches(op[1], options.partialrecursive):
                    matched_self_i.append(sp)
                    matched_other_i.append(op)
                    break
        for sp in [p for p in self.products if p not in matched_self_i]:
            self_only_args.append('Product' + sp[0])
        for op in [p for p in e.products if p not in matched_other_i]:
            other_only_args.append('Product' + op[0])

        # determine Theme/Site matches.

        # if "theme only" is specified, drop sites on local
        # copies of the lists for the comparison.
        if match_theme_only:
            self_theme_sites = [(x[0], x[1], None) for x in self.theme_sites]
            other_theme_sites = [(x[0], x[1], None) for x in e.theme_sites]
        else:
            self_theme_sites = self.theme_sites[:]
            other_theme_sites = e.theme_sites[:]

        # to simplify processing, first determine full Theme/Site pair
        # matches.
        matched_self_ts, matched_other_ts = [], []
        for sts in self_theme_sites:
            for ots in other_theme_sites:

                if ots in matched_other_ts:
                    continue  # only match once

                if ((sts[1] is None and ots[1] is None) or
                        (sts[1] is not None and
                         sts[1].matches(ots[1], options.partialrecursive))):
                    # theme match
                    if ((sts[2] is None and ots[2] is None) or
                            (sts[2] is not None and sts[2].matches(ots[2], options.partialrecursive))):
                        # site match
                        matched_self_ts.append(sts)
                        matched_other_ts.append(ots)
                        break

        unmatched_self_ts = [st for st in self_theme_sites if st not in matched_self_ts]
        unmatched_other_ts = [st for st in other_theme_sites if st not in matched_other_ts]

        # partial match still possible.

        # analyse remaining: we can still have Theme matches with
        # differences in Sites.
        theme_matched_self_ts, theme_matched_other_ts = [], []
        self_only_ts, other_only_ts = [], []
        for sts in unmatched_self_ts:
            match_found = False
            for ots in unmatched_other_ts:
                if sts[1].matches(ots[1], options.partialrecursive):
                    theme_matched_self_ts.append(sts)
                    theme_matched_other_ts.append(ots)
                    match_found = True
                    break
            if not match_found:
                self_only_ts.append(sts)
        other_only_ts = [ots for ots in unmatched_other_ts if ots not in theme_matched_other_ts]

        # break Theme/Site pairs into their constituents
        # (this is horribly badly done, clean up)
        for st in matched_self_ts:
            if st[1] is not None:
                self_matched_args.append("Theme%s" % st[0])
            if st[2] is not None:
                self_matched_args.append("Site%s" % st[0])
        for st in matched_other_ts:
            if st[1] is not None:
                other_matched_args.append("Theme%s" % st[0])
            if st[2] is not None:
                other_matched_args.append("Site%s" % st[0])

        for st in theme_matched_self_ts:
            if st[1] is not None:
                self_matched_args.append("Theme%s" % st[0])
            if st[2] is not None:
                self_only_args.append("Site%s" % st[0])
        for st in theme_matched_other_ts:
            if st[1] is not None:
                other_matched_args.append("Theme%s" % st[0])
            if st[2] is not None:
                other_only_args.append("Site%s" % st[0])

        for st in self_only_ts:
            if st[1] is not None:
                self_only_args.append("Theme%s" % st[0])
            if st[2] is not None:
                self_only_args.append("Site%s" % st[0])
        for st in other_only_ts:
            if st[1] is not None:
                other_only_args.append("Theme%s" % st[0])
            if st[2] is not None:
                other_only_args.append("Site%s" % st[0])

        # decide.
        if match_theme_only:
            # discard differences not relating to theme
            self_only_args = [a for a in self_only_args if a[:5] == "Theme"]
            other_only_args = [a for a in other_only_args if a[:5] == "Theme"]

        if match_partial:
            return len(other_only_args) == 0
        else:
            # normal
            return len(self_only_args) == 0 and len(other_only_args) == 0


# represents event modification annotation
class Modification:
    def __init__(self, id, type, eid):
        self.id, self.type, self.eid = id, type, eid
        self.event = None

    def resolve_idrefs(self, annotation_by_id):
        assert self.eid in annotation_by_id, "ERROR: undefined ID %s referenced in %s" % (self.eid, self.id)
        self.event = annotation_by_id[self.eid]
        assert self.event.type in event_types, "ERROR: non-Event %s referenced in %s" % (self.eid, self.id)


# represents a modification (negation or speculation) pseudo-event.
class ModificationEvent(Event):
    def __init__(self, mod):
        self.id, self.type, self.eid, self.event = mod.id, mod.type, mod.eid, mod.event

    def matches(self, e, match_theme_only=False, match_partial=False):
        global options

        if not isinstance(e, ModificationEvent):
            return False

        if self.type != e.type:
            return False

        return self.event.matches(e.event, options.partialrecursive, match_partial)

    def to_string(self):
        return "%s\t%s %s" % (self.id, self.type, self.eid)


# parses and verifies the given text line as text-bound annotation
# (entity/event trigger), returns Textbound object.
def parse_textbound_line(l, fn):
    # three tab-separated fields
    fields = l.split("\t")
    id, annotation, text = None, None, None
    # allow two or three fields, with the last (reference text) allowed to be empty.
    if len(fields) == 3:
        id, annotation, text = fields
    elif len(fields) == 2:
        id, annotation = fields
        text = ""
    else:
        assert False, "FORMAT ERROR: unexpected number (%d) of tab-separated fields: line '%s' in %s" % (
        len(fields), l, fn)

    # id is usually in format "GT[0-9]+", but may have other simiarl
    if not re.match(r'^[GT]\d+$', id):
        # print >> sys.stderr, "NOTE: ID not in format '[GT][0-9]+': line '%s' in %s" % (l, fn)
        pass
    assert re.match(r'^[A-Z]\d+$', id), "FORMAT ERROR: ID not in format '[GT][0-9]+': line '%s' in %s" % (l, fn)

    # annotation is three space-separated fields
    fields = re.split(' +', annotation)
    # print("parse_textbound_line fields", fields)
    assert len(fields) == 3, "FORMAT ERROR: unexpected number of space-separated fields: line '%s' in %s" % (l, fn)
    type, start, end = fields
    assert type in textbound_types, "FORMAT ERROR: disallowed type for text-bound annotation: line '%s' in %s" % (l, fn)

    # start and end are offsets into the text
    assert re.match(r'^\d+$', start) and re.match(r'^\d+$',
                                                  end), "FORMAT ERROR: offsets not in '[0-9]+' format: line '%s' in %s" % (
    l, fn)
    start, end = int(start), int(end)
    assert start < end, "FORMAT ERROR: start offset not smaller than end offset: line '%s' in %s" % (l, fn)

    return Textbound(id, type, start, end, text)


# parses and verifies the given text line as an given entity (Protein
# etc.) annotation, returns Textbound object.
def parse_given_entity_line(l, fn, document_text):
    global options

    l = l.strip(' \n\r')

    e = parse_textbound_line(l, fn)
    if options.verifytext:
        e.verify_text(document_text, fn)
    assert e.type in given_types, "FORMAT ERROR: non-given type not allowed here: line '%s' in %s" % (l, fn)

    # OK, checks out.
    return e


# parses and verifies the given text line as event annotation,
# return Event object.
def parse_event_line(l, fn):
    global options

    # special case for "alignment" work: allow line-terminal space followed
    # by "comment"
    l = re.sub(r'\s*\#.*', '', l)

    # two tab-separated fields
    fields = l.split("\t")
    assert len(fields) == 2, "FORMAT ERROR: unexpected number of tab-separated fields: line '%s' in %s" % (l, fn)
    id, annotation = fields

    # id must be in format "E[0-9]+" (unless split eval)
    if not options.spliteventeval:
        assert re.match(r'^E\d+$', id), "FORMAT ERROR: ID not in format 'E[0-9]+': line '%s' in %s" % (l, fn)

    # annotation is two or more space-separated fields
    fields = re.split(' +', annotation)
    # exception: there may be no-argument events; in this case fill in
    # a fake second field
    if len(fields) == 1:
        fields.append("")
    assert len(fields) >= 2, "FORMAT ERROR: unexpected number of space-separated fields: line '%s' in %s" % (l, fn)
    type_tid, args = fields[0], fields[1:]

    # first field is "TYPE:ID", where ID is for the event trigger
    parts = type_tid.split(":")
    assert len(parts) == 2, "FORMAT ERROR: event type not in 'TYPE:ID' format: line '%s' in %s" % (l, fn)
    type, tid = parts
    assert type in event_types, "FORMAT ERROR: disallowed type for event annotation: line '%s' in %s" % (l, fn)
    assert re.match(r'^T\d+$', tid), "FORMAT ERROR: event trigger ID not in format 'T[0-9]+': line '%s' in %s" % (l, fn)

    # each argument is "ARG:ID", where ID is either for an event or text-bound.
    split_args = []
    for a in args:
        # for no-argument events
        if a == "":
            continue
        parts = a.split(":")
        assert len(parts) == 2, "FORMAT ERROR: argument type not in 'ARG:ID' format: line '%s' in %s" % (l, fn)
        arg, aid = parts
        # Note: workarounds for data issues; fix data and remove
        if arg == 'Cause2':
            # print >> sys.stderr, 'Note: reading "Cause2" as "Cause"'
            arg = 'Cause'
        if type == 'Pathway' and arg.startswith('Theme'):
            # print >> sys.stderr, 'Note: reading Pathway "Theme" as "Participant"'
            arg = re.sub(r'^Theme', 'Participant', arg)
        assert valid_argument_name(arg), "FORMAT ERROR: invalid argument name: line '%s' in %s" % (l, fn)
        assert re.match(r'^[TE]\d+', aid), "FORMAT ERROR: invalid id in argument: line '%s' in %s" % (l, fn)
        # less restricted prefix check
        # assert re.match(r'^[A-Z]\d+', aid), "FORMAT ERROR: invalid id in argument: line '%s' in %s" % (l, fn)
        split_args.append((arg, aid))

    return Event(id, type, tid, split_args)


# parses and verifies the given text line as modification annotation,
# returns Modification object.
def parse_modification_line(l, fn):
    global options

    # two tab-separated fields
    fields = l.split("\t")
    assert len(fields) == 2, "FORMAT ERROR: unexpected number of tab-separated fields: line '%s' in %s" % (l, fn)
    id, annotation = fields

    # id must be in format "M[0-9]+" (unless split eval)
    if not options.spliteventeval:
        assert re.match(r'^M\d+$', id), "FORMAT ERROR: ID not in format 'M[0-9]+': line '%s' in %s" % (l, fn)

    # annotation is two space-separated fields
    fields = re.split(' +', annotation)
    # print("parse_modification_line fields", fields)
    assert len(fields) == 2, "FORMAT ERROR: unexpected number of space-separated fields: line '%s' in %s" % (l, fn)
    type, eid = fields
    assert type in modification_types, "FORMAT ERROR: disallowed type for modification annotation: line '%s' in %s" % (
    l, fn)

    return Modification(id, type, eid)


# parses and verifies an "Equiv" line, returns Equiv object.
def parse_equiv_line(l, fn):
    # two tab-separated fields
    fields = l.split("\t")
    assert len(fields) == 2, "FORMAT ERROR: unexpected number of tab-separated fields: line '%s' in %s" % (l, fn)
    id, annotation = fields

    # "id" must be "*"
    assert id == "*", "FORMAT ERROR: invalid ID field: line '%s' in %s" % (l, fn)

    # annotation is three or more space-separated fields
    fields = re.split(' +', annotation)
    assert len(fields) >= 2, "FORMAT ERROR: unexpected number of space-separated fields: line '%s' in %s" % (l, fn)
    type, ids = fields[0], fields[1:]
    assert type == "Equiv", "FORMAT ERROR: non-'Equiv' type for '*' id: line '%s' in %s" % (l, fn)

    # IDs are in the "T[0-9]+" format.
    for id in ids:
        assert re.match(r'^[A-Z]\d+$', id), "FORMAT ERROR: IDs not in format 'T[0-9]+': line '%s' in %s" % (l, fn)

    # IDs should be unique; allow but ignore other
    seen_ids = set()
    unique_ids = []
    for id in ids:
        if id in seen_ids:
            print(sys.stderr, "Note: id '%s' appears multiple times in Equiv '%s', ignoring repetition" % (id, l))
        else:
            seen_ids.add(id)
            unique_ids.append(id)
    ids = unique_ids

    return Equiv(ids)


def open_annotation_file(fn):
    try:
        f = open(fn)
    except IOError as e:
        usage_exit("error: " + str(e) + " (try '-r' option?)")
    return f


def check_unique_and_store(key, val, map, fn):
    assert key not in map, "ERROR: duplicate ID %s in %s" % (key, fn)
    map[key] = val


# reads in and verifies event annotation (".a2 file") from the given
# file, with reference to the given original document text.
# Last argument is a map (id to annotation obj) of previously read
# annotations.  Returns a pair of (annotation_map, equiv_list) where
# annotation_map is a map from ID to the corresponding annotation and
# equiv_list is a list of Equiv's.
def parse_event_file(a2file, document_text, annotation_by_id):
    global options

    equiv_annotations = []

    for l in a2file:
        l = l.strip(' \n\r')

        # decide how to parse this based on first character of ID
        if len(l) > 0 and l[0] == "R":
            continue
        assert len(l) > 0 and l[0] in ("*", "E", "M", "T", "G", "L", "D", "P", "A",
                                       "N"), "FORMAT ERROR: line doesn't start with valid ID: line '%s' in %s" % (
        l, a2file.name)

        # parse and store as appropriate.
        if l[0] == "*":
            equiv = parse_equiv_line(l, a2file.name)
            equiv_annotations.append(equiv)
        elif l[0] in ("T", "G", "L", "D", "P", "A", "N"):
            t = parse_textbound_line(l, a2file.name)
            if options.verifytext:
                t.verify_text(document_text, a2file.name)
            check_unique_and_store(t.id, t, annotation_by_id, a2file.name)
        elif l[0] == "E":
            # special case: generate-task-specific-a2-file.pl may
            # create events with an empty set of arguments in
            # processing "split event" data. In "split event" mode, ignore
            # these without error.
            if options.spliteventeval and re.match(r'^E\S+\t[A-Za-z_]+:T\d+\s*$', l):
                continue

            e = parse_event_line(l, a2file.name)
            check_unique_and_store(e.id, e, annotation_by_id, a2file.name)
        elif l[0] == "M":
            m = parse_modification_line(l, a2file.name)
            check_unique_and_store(m.id, m, annotation_by_id, a2file.name)
        else:
            assert False, "INTERNAL ERROR"

    return annotation_by_id, equiv_annotations


# helper for a special circumstance in "split event" evaluation:
# recursively removes all events that fail idref resolutions (i.e.
# refer to an event that doesn't exist)
def remove_idref_failing_events(annotation_by_id, fn):
    while True:
        dropped_event = False
        annids = annotation_by_id.keys()
        for i in annids:
            a = annotation_by_id[i]
            try:
                a.resolve_idrefs(annotation_by_id)
            except (AssertionError, e):
                print(sys.stderr, "Dropping event %s in %s:" % (i, fn), str(e))
                del annotation_by_id[i]
                dropped_event = True
        if not dropped_event:
            return


# reads in and verifies the reference files. Returns tuples
# (document_text, annotation_map, equiv_list) where document_text is
# the raw unannotated text of the original document, annotation_map is
# a map from ID to the corresponding annotation objects and equiv_list
# is a list of Equiv annotation objects.
#
# Argument t2_tasksuffix gives a suffix to append to the ".a2"
# filename from which event annotations are read. So, to e.g. evaluate
# specifically against task 1 reference files, provide ".t1".
def parse_reference_files(PMID, t2_tasksuffix=""):
    # store all other annotations by id, Equivs as a list (no id)
    annotation_by_id = {}

    # read text
    txt_fn = os.path.join(options.refdir, PMID + ".txt")
    txt_f = open_annotation_file(txt_fn)

    document_text = txt_f.read()
    txt_f.close()

    # read given entities (".a1 file").
    a1_fn = os.path.join(options.refdir, PMID + ".a1")
    a1_f = open_annotation_file(a1_fn)

    for l in a1_f:
        e = parse_given_entity_line(l, a1_fn, document_text)
        check_unique_and_store(e.id, e, annotation_by_id, a1_fn)
    a1_f.close()

    # read other annotations (".a2 file"). The file to read may be modified by
    # a task suffix if provided.

    # exception: the ".unified" suffix should be ignored
    if t2_tasksuffix != ".unified":
        a2_fn = os.path.join(options.refdir, PMID + ".a2" + t2_tasksuffix)
    else:
        a2_fn = os.path.join(options.refdir, PMID + ".a2")
    a2_f = open_annotation_file(a2_fn)
    annotation_by_id, equiv_annotations = parse_event_file(a2_f, document_text, annotation_by_id)
    a2_f.close()

    # now that we have all the annotations, check and resolve ID
    # references. Exception for "split event" evaluation: allow broken
    # idrefs (these are sometimes generated in processing ...  a bit
    # of a long story) and just remove the events.
    if options.spliteventeval:
        remove_idref_failing_events(annotation_by_id, a2_fn)

    for a in annotation_by_id.values():
        a.resolve_idrefs(annotation_by_id)

    for e in equiv_annotations:
        e.resolve_idrefs(annotation_by_id)

    # then, convert Modifications into events: Negation and Speculation are
    # converted into pseudo-Events for the purposes of matching.
    pseudoEvents = {}
    for a in [e for e in annotation_by_id.values() if e.type == "Negation" or e.type == "Speculation"]:
        pseudoEvents[a.id] = ModificationEvent(a)
    # replace the originals
    for id in pseudoEvents:
        annotation_by_id[id] = pseudoEvents[id]

    # finally, mark entities with their equivalent entities.
    for p in [e for e in annotation_by_id.values() if e.type in entity_types]:
        p.equivs = []
    for e in equiv_annotations:
        for p in e.entities:
            assert p.equivs == [], "ERROR: multiple Equivs for %s" % p.id
            p.equivs = [o for o in e.entities if o != p]

    return (document_text, annotation_by_id, equiv_annotations)


# returns PMID from ".a2" file name
def a2file_PMID(fn):
    bfn = os.path.basename(fn)
    # NOTE: not just "PMID" anymore, allow full text format also.
    m = re.match(r'^((?:PMID-|PMC-?)?\d+(?:\d+-.*)?)\.a2(?:.core)?', bfn)
    if not m:
        usage_exit("error: filename '%s' not in expected format" % bfn)
    return m.group(1)


def usage_exit(msg=None):
    global op
    op.print_help()
    if msg is not None:
        # print >> sys.stderr, "\n"+msg
        print(sys.stderr, "\n" + msg)
    sys.exit(1)


# parse command-line arguments
op = optparse.OptionParser(
    "\n  %prog [OPTIONS] FILES\n\nDescription:\n  Evaluates performance for given BioNLP Shared Task .a2 files.")
op.add_option("-r", "--ref_directory", action="store", dest="refdir", metavar="DIRECTORY", default=".",
              help="the directory where reference files are placed.")
op.add_option("-d", "--pred_directory", action="store", dest="preddir", metavar="DIRECTORY", default=".",
              help="the directory where predicted files are placed.")
# op.add_option("-t","--task",action="store",dest="task",metavar="1,2,3",default=None,help="the number of the task for which to evaluate.")
op.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False, help="verbose output.")
op.add_option("-s", "--softboundary", action="store_true", dest="softboundary", default=False,
              help="\"soft\" matching for textbound entity boundaries.")
op.add_option("-p", "--partialrecursive", action="store_true", dest="partialrecursive", default=False,
              help="partial recursive matching of event arguments.")
op.add_option("-1", "--singlepenalty", action="store_true", dest="singlepenalty", default=False,
              help="single penalty for partial matches.")
op.add_option("-t", "--verifytext", action="store_true", dest="verifytext", default=False,
              help="require that correct texts are given for textbound annotations.")
op.add_option("-S", "--spliteval", action="store_true", dest="spliteventeval", default=False,
              help="Evaluate in \"split events\" mode (relaxes constraints)")
op.add_option("-o", "--output_directory", action="store", dest="outputdir", metavar="DIRECTORY", default=None,
              help="the directory where standoff files with evaluation results are placed.")

options, args = op.parse_args()

import os

args = [options.preddir + file_name for file_name in os.listdir(options.preddir) if
        os.path.isfile(os.path.join(options.preddir, file_name)) and file_name.endswith(".a2")]

# predDir = "output/pred/"
# directories = os.listdir(options.preddir)
# for f in directories:
#    args.append(f)

# discard args files with unknown suffixes.
kept_args = []
# known_suffixes = set(["a2", "core"])
known_suffixes = set(["a2"])
if options.spliteventeval:
    # allow an added "a" or "s" to any suffix
    known_suffixes |= set([s + "a" for s in known_suffixes] +
                          [s + "s" for s in known_suffixes])

# print("args", len(args))
for a in args:
    if a.split(".")[-1] not in known_suffixes:
        # print >> sys.stderr, "Skipping file %s: unknown suffix" % a
        print(sys.stderr, "Skipping file %s: unknown suffix" % a)
    else:
        kept_args.append(a)
# print("kept_args", kept_args)
args = kept_args

if args == []:
    usage_exit("error: no recognized FILES given.")

# figure out which task we're targeting, determining the
# task suffix to use for the reference files.
lastsuffixes = set([a.split(".")[-1] for a in args])
if len(lastsuffixes) > 1:
    usage_exit("error: given files don't all share the task suffix (%s given)" % ",".join(lastsuffixes))
lastsuffix = list(lastsuffixes)[0]
# if lastsuffix not in ("t1", "t12"):
#    usage_exit("error: evaluation for '.%s' files not supported (SORRY!)" % lastsuffix)
reference_t2_tasksuffix = "." + lastsuffix

# exception: if ".a2" files are provided, use an empty "tasksuffix"
if reference_t2_tasksuffix == ".a2":
    reference_t2_tasksuffix = ""

# exception: if ".core" files are provided, use an empty "tasksuffix"
if reference_t2_tasksuffix == ".core":
    reference_t2_tasksuffix = ""

# verify arguments, store PMIDs of files to check.
PMIDs = []
for a in args:
    PMID = a2file_PMID(a)
    if PMID in PMIDs:
        usage_exit("error: filename with PMID %s given multiple times." % PMID)
    PMIDs.append(PMID)

# get gold, store (text, ann_by_id, equiv) tuples indexed by PMID.
gold_annotation = {}
for PMID in PMIDs:
    # print("pmid", PMID)
    # print("reference_t2_tasksuffix", reference_t2_tasksuffix)
    try:
        document_text, gold_ann_by_id, gold_equiv = parse_reference_files(PMID, reference_t2_tasksuffix)
        # print("pmid", PMID)
    except AssertionError as e:
        # print >> sys.stderr, "ERROR: Failed to read gold annotation for PMID (skipping)", PMID, ":", e
        print(sys.stderr, "ERROR: Failed to read gold annotation for PMID (skipping)", PMID, ":", e)
        continue

    assert PMID not in gold_annotation, "INTERNAL ERROR"
    gold_annotation[PMID] = (document_text, gold_ann_by_id, gold_equiv)

# process the files to evaluate, store by PMID.
annotation = {}
answer_equiv = {}
for fn in args:
    PMID = a2file_PMID(fn)

    # skip if reading gold failed
    if PMID not in gold_annotation:
        continue

    # get gold for reference
    document_text, gold_ann_by_id, gold_equiv = gold_annotation[PMID]

    try:
        f = open(fn)
    except IOError as e:
        print("error here")
        usage_exit("error: " + str(e))

    # the non-gold annotation can refer to gold Protein annotations
    # and cannot reuse gold Protein IDs.
    ann_by_id = {}
    for id, a in gold_ann_by_id.items():
        if a.type in given_types:
            ann_by_id[id] = a

    try:
        ann_by_id, equiv = parse_event_file(f, document_text, ann_by_id)

        # special case for "split event" evaluation. See similar case
        # in gold data read.
        if options.spliteventeval:
            remove_idref_failing_events(ann_by_id, fn)

        for a in ann_by_id.values():
            # print("a:", a)
            # print("ann_by_id:", ann_by_id)
            a.resolve_idrefs(ann_by_id)

        # Negation and Speculation
        pseudoEvents = {}
        for a in [e for e in ann_by_id.values() if e.type == "Negation" or e.type == "Speculation"]:
            pseudoEvents[a.id] = ModificationEvent(a)
        # replace the originals
        for id in pseudoEvents:
            ann_by_id[id] = pseudoEvents[id]

    # except (AssertionError, e):
    except AssertionError as e:
        print(sys.stderr, "Failed to read", fn, ":", str(e))
        continue

    f.close()

    assert PMID not in annotation, "INTERNAL ERROR"
    annotation[PMID] = ann_by_id

    if len(equiv) != 0:
        answer_equiv[PMID] = equiv

if len(answer_equiv) != 0:
    print(sys.stderr, "Note: answer contained 'Equiv' annotation. This is ignored.")

# Remove "simple" duplicate events from the annotation. This has to be
# done separately from the "approximate matching" duplicate
# elimination below, as that is based on gold matches and cannot
# eliminate FP duplicates.

# during this processing, maintain strict matching
tmp_softboundary, tmp_partialrecursive, tmp_singlepenalty = options.softboundary, options.partialrecursive, options.singlepenalty
options.softboundary, options.partialrecursive, options.singlepenalty = False, False, False

for PMID in annotation:
    ann_by_id = annotation[PMID]

    seen_events = []
    dup_event_ids = []

    for id, e in ann_by_id.items():
        if id[0] not in ("E", "M"):
            continue
        dup_found = False
        for se in seen_events:
            if se.matches(e):
                # print >> sys.stderr, "Note: PMID %s: %s and %s are identical, ignoring %s." % (PMID, se.id, id, id)
                dup_found = True
                break
        if dup_found:
            dup_event_ids.append(id)
        seen_events.append(e)

    for id in dup_event_ids:
        if options.verbose:
            print(sys.stderr, "duplicate: %s" % (PMID + "-" + id))
        del ann_by_id[id]
    annotation[PMID] = ann_by_id

# restore matching options
options.softboundary, options.partialrecursive, options.singlepenalty = tmp_softboundary, tmp_partialrecursive, tmp_singlepenalty

# If soft boundary matching is specified, calculate extended start and
# end points for all gold Textbound entities.
if options.softboundary:
    for PMID in annotation:
        if PMID not in gold_annotation or PMID not in annotation:
            continue

        document_text, gold_ann_by_id, gold_equiv = gold_annotation[PMID]
        ann_by_id = annotation[PMID]

        gold_textbound = [t for t in gold_ann_by_id.values() if isinstance(t, Textbound)]
        textbound = [t for t in ann_by_id.values() if isinstance(t, Textbound)]

        # extension is not allowed to cause the extended region to
        # overlap with another gold Textbound. For checking this,
        # mark all characters in Textbound spans as reserved.
        reserved_index = {}
        for tb in gold_textbound:
            for c in range(tb.start, tb.end):
                # simple
                # reserved_index[c] = 1

                # debug version
                if c not in reserved_index:
                    reserved_index[c] = []
                reserved_index[c].append(tb.id)

        for tb in gold_textbound:
            tb.identify_extended_span(document_text, reserved_index)

# Remove "approximate matching" duplicate events. This is a bit more
# involved: two events are "approximate matching" duplicates if they
# match the exact same non-empty set of events according to the
# matching criteria. Additionally, the duplicate elimination should
# not reduce the number to less than that of the matched gold events.

for PMID in annotation:
    if PMID not in gold_annotation or PMID not in annotation:
        continue

    document_text, gold_ann_by_id, gold_equiv = gold_annotation[PMID]
    ann_by_id = annotation[PMID]

    gold_events = [e for e in gold_ann_by_id.values() if isinstance(e, Event)]
    events = [e for e in ann_by_id.values() if isinstance(e, Event)]

    # keep track with dict indexed by tuples of ids of matched gold
    # events, values lists of ids of matching answer events.
    events_by_matched = {}

    for e in events:
        matched_gold_ids = []
        for ge in gold_events:
            if e.matches(ge):
                matched_gold_ids.append(ge.id)

        # unmatched can't be "approximate matching" dups
        if matched_gold_ids == []:
            continue

        match_tuple = tuple(sorted(matched_gold_ids))
        if match_tuple not in events_by_matched:
            events_by_matched[match_tuple] = []
        events_by_matched[match_tuple].append(e.id)

    # in each case where the number of matching events is larger than the
    # number of matched gold events, remove "extra" duplicates.
    for mt in events_by_matched:
        matched_gold = list(mt)
        matching = events_by_matched[mt]
        if len(matching) > len(matched_gold):
            toremove = matching[len(matched_gold):]
            # print >> sys.stderr, "Note: events %s in %s are equivalent to gold event(s) %s under partial matching criteria. Discarding %s." % (",".join(matching), PMID, ",".join(matched_gold), ",".join(toremove))
            for id in toremove:
                if options.verbose:
                    print(sys.stderr, "duplicate: %s" % (PMID + "-" + id))
                del ann_by_id[id]

    annotation[PMID] = ann_by_id

########################################
#
# scoring starts here.
#
########################################

# note variant of TP/FP/FN scheme: as an answer event can match
# multiple gold events in soft matching, we'll maintain match
# counts separately for gold (TPg) and the answer (TPa). Then
# prec = TPa / (TPa+FP) and
# rec  = TPg / (TPg+FN).
# (Below, TPa and TPg are "match_answer" and "match_gold".)

# If partial matching is applied, the scheme is further modified to
# count cases where an answer event matches a subset of a gold event
# arguments (P) and where a gold event matches a subset of answer
# event arguments (O). Answer events are then classified into
# TPa/P/FP (each event into exactly one class) and gold events into
# TPg/O/FN (likewise). For calculating prec/rec/F, P count as
# neither TPa nor FP, and O count as both TPg and FP.

total_match_answer, total_match_gold, total_false_positive, total_false_negative = 0, 0, 0, 0
total_partial_match, total_over_match = 0, 0
total_type_match_answer, total_type_match_gold, total_type_false_positive, total_type_false_negative = {}, {}, {}, {}
total_type_partial_match, total_type_over_match = {}, {}
for t in event_types | modification_types:
    total_type_match_answer[t] = total_type_match_gold[t] = total_type_false_positive[t] = total_type_false_negative[
        t] = 0
    total_type_partial_match[t] = total_type_over_match[t] = 0

# need to remember these per document for stats file output
doc_match_answer, doc_match_gold, doc_false_positive, doc_false_negative = {}, {}, {}, {}
doc_partial_match, doc_over_match = {}, {}

number_of_documents_evaluated = 0

for PMID in PMIDs:
    # skip if reading gold or given annotation failed
    if PMID not in gold_annotation or PMID not in annotation:
        # print >> sys.stderr, "#"*78
        print(sys.stderr, "#" * 78)

        if PMID not in gold_annotation:
            print("PMID not in gold:", PMID)
        if PMID not in annotation:
            print("PMID not in annotation:", PMID)
        # print >> sys.stderr, "# NOTE: SKIPPING %s, READ FAILED: RESULTS WILL NOT BE VALID FOR THE FULL DATASET!" % fn
        print(sys.stderr, "# NOTE: SKIPPING %s, READ FAILED: RESULTS WILL NOT BE VALID FOR THE FULL DATASET!" % fn)
        # print >> sys.stderr, "#"*78
        print(sys.stderr, "#" * 78)
        continue

    document_text, gold_ann_by_id, gold_equiv = gold_annotation[PMID]
    ann_by_id = annotation[PMID]

    gold_events = [e for e in gold_ann_by_id.values() if isinstance(e, Event)]
    events = [e for e in ann_by_id.values() if isinstance(e, Event)]

    # mark all gold and given events non-matched
    for e in gold_events + events:
        e.clear_matching()

    # then go through given events and try to match with gold events. Note that
    # one answer may match multiple gold events (soft matching).

    for e in events:
        for ge in gold_events:
            if not options.singlepenalty:
                # standard direct match
                if e.matches(ge):
                    ge.mark_matching(e)
                    e.mark_matching(ge)
            else:
                # two-way match, checking partials
                part = ge.matches(e, match_partial=True)
                over = e.matches(ge, match_partial=True)
                if part and over:
                    # subset and superset -> equal. sanity check
                    assert e.matches(ge, match_partial=False), "INTERNAL ERROR: two partial matches but no full match"
                    ge.mark_matching(e)
                    e.mark_matching(ge)
                elif part:
                    e.mark_partially_matched(ge)
                    ge.mark_partially_matched_by(e)
                elif over:
                    ge.mark_partially_matched(e)
                    e.mark_partially_matched_by(ge)

    match_answer, match_gold, false_positive, false_negative = 0, 0, 0, 0
    partial_match, over_match = 0, 0
    type_match_answer, type_match_gold, type_false_positive, type_false_negative = {}, {}, {}, {}
    type_partial_match, type_over_match = {}, {}

    for t in event_types | modification_types:
        type_match_answer[t] = type_match_gold[t] = type_false_positive[t] = type_false_negative[t] = 0
        type_partial_match[t] = type_over_match[t] = 0

    # take counts
    for e in events:
        if e.has_matched():
            match_answer += 1
            type_match_answer[e.type] += 1
        elif e.has_partially_matched():
            partial_match += 1
            type_partial_match[e.type] += 1
        else:
            false_positive += 1
            type_false_positive[e.type] += 1

        if options.verbose:
            if e.has_matched():
                # print >> sys.stderr, "match %s:%s : %s" % (e.type, PMID+'-'+e.id, ",".join([PMID+'-'+x.id for x in e.matching]))
                print(sys.stderr, "match %s:%s : %s" % (
                e.type, PMID + '-' + e.id, ",".join([PMID + '-' + x.id for x in e.matching])))
            elif e.has_partially_matched():
                # print >> sys.stderr, "part. %s:%s : %s" % (e.type, PMID+'-'+e.id, ",".join([PMID+'-'+x.id for x in e.partially_matched]))
                print(sys.stderr, "part. %s:%s : %s" % (
                e.type, PMID + '-' + e.id, ",".join([PMID + '-' + x.id for x in e.partially_matched])))
            else:
                # print >> sys.stderr, "f.pos %s:%s" % (e.type, PMID+'-'+e.id)
                print(sys.stderr, "f.pos %s:%s" % (e.type, PMID + '-' + e.id))

    for ge in gold_events:
        if ge.has_matched():
            match_gold += 1
            type_match_gold[ge.type] += 1
        elif ge.has_partially_matched():
            over_match += 1
            type_over_match[ge.type] += 1
        else:
            false_negative += 1
            type_false_negative[ge.type] += 1

        if options.verbose:
            if ge.has_partially_matched():
                # print >> sys.stderr, "over %s:%s : %s" % (ge.type, PMID+'-'+ge.id, ",".join([PMID+'-'+x.id for x in ge.partially_matched]))
                print(sys.stderr, "over %s:%s : %s" % (
                ge.type, PMID + '-' + ge.id, ",".join([PMID + '-' + x.id for x in ge.partially_matched])))
            elif not ge.has_matched():
                # print >> sys.stderr, "f.neg %s:%s" % (ge.type, PMID+'-'+ge.id)
                print(sys.stderr, "f.neg %s:%s" % (ge.type, PMID + '-' + ge.id))

    # sanity
    assert match_gold + over_match + false_negative == len(gold_events), "INTERNAL ERROR"
    assert match_answer + partial_match + false_positive == len(events), "INTERNAL ERROR"
    assert sum(type_match_gold.values()) == match_gold, "INTERNAL ERROR"
    assert sum(type_over_match.values()) == over_match, "INTERNAL ERROR"
    assert sum(type_match_answer.values()) == match_answer, "INTERNAL ERROR"
    assert sum(type_false_positive.values()) == false_positive, "INTERNAL ERROR"
    assert sum(type_partial_match.values()) == partial_match, "INTERNAL ERROR"
    assert sum(type_false_negative.values()) == false_negative, "INTERNAL ERROR"

    total_match_gold += match_gold
    total_match_answer += match_answer
    total_partial_match += partial_match
    total_over_match += over_match
    total_false_positive += false_positive
    total_false_negative += false_negative

    for t in event_types | modification_types:
        total_type_match_gold[t] += type_match_gold[t]
        total_type_match_answer[t] += type_match_answer[t]
        total_type_partial_match[t] += type_partial_match[t]
        total_type_over_match[t] += type_over_match[t]
        total_type_false_positive[t] += type_false_positive[t]
        total_type_false_negative[t] += type_false_negative[t]

    # for stats file output

    doc_match_gold[PMID] = match_gold
    doc_match_answer[PMID] = match_answer
    doc_partial_match[PMID] = partial_match
    doc_over_match[PMID] = over_match
    doc_false_positive[PMID] = false_positive
    doc_false_negative[PMID] = false_negative

    number_of_documents_evaluated += 1

##########
#
# "Merged" standoff output (if specified)

if options.outputdir is not None:
    # comments should be ID-like, keep idx
    next_free_comment_idx = 1

    for PMID in PMIDs:
        # skip if reading gold or given annotation failed
        if PMID not in gold_annotation or PMID not in annotation:
            continue

        document_text, gold_ann_by_id, gold_equiv = gold_annotation[PMID]
        ann_by_id = annotation[PMID]

        # for output combining events from gold and submission,
        # non-given IDs must be revised to avoid clashes. Also,
        # references to non-given textbounds matching gold must be
        # revised to avoid duplication. For the former, simply attach
        # a fixed string to all submission IDs.

        submission_textbound = [a for a in ann_by_id.values() if isinstance(a, Textbound)]
        gold_textbound = [a for a in gold_ann_by_id.values() if isinstance(a, Textbound)]

        # First, not all matching textbounds are necessarily compared
        # (due to e.g. differences in referencing event type), so
        # "matching" is not exhaustively filled. Complete by
        # exhaustive comparison.

        for t in submission_textbound:
            if not t.matching:
                for g in gold_textbound:
                    t.matches(g)

        # keep note of which textbound references could not be mapped
        referenced_unmappable = {}

        # for easy reference, create a mapping from submission
        # textbound IDs to matched gold IDs.
        textbound_gold_map = {}
        for t in submission_textbound:
            if t.matching:
                # just pick the first matched (a bit arbitrary)
                textbound_gold_map[t.id] = t.matching[0].id

        submission_id_suffix = "-submission"
        revised_ann_by_id = {}
        for aid, ann in ann_by_id.items():
            # skip givens (in gold)
            if ann.type in given_types:
                continue

            newid = aid + submission_id_suffix
            # sorry
            if isinstance(ann, Textbound):
                ann.id = newid
            elif isinstance(ann, ModificationEvent):
                ann.id = newid
                # if the modified event matches gold event(s), map to
                # the first matched; otherwise revise referenced ID
                if ann.event.has_matched():
                    ann.event, ann.eid = ann.event.matching[0], ann.event.matching[0].id
                else:
                    ann.eid = ann.eid + submission_id_suffix
            elif isinstance(ann, Event):
                ann.id = newid
                if ann.tid in textbound_gold_map:
                    ann.tid = textbound_gold_map[ann.tid]
                    ann.trigger = gold_ann_by_id[ann.tid]
                else:
                    ann.tid = ann.tid + submission_id_suffix
                    referenced_unmappable[ann.tid] = True
                newargs = []
                for arg, argid in ann.args:
                    # references to given gold annotation need not be revised;
                    # references to annotation matching gold can be mapped,
                    # others should have IDs revised.
                    if argid in gold_ann_by_id and gold_ann_by_id[argid].type in given_types:
                        newargs.append((arg, argid))
                    elif argid in textbound_gold_map:
                        newargs.append((arg, textbound_gold_map[argid]))
                    elif argid in ann_by_id and ann_by_id[argid].type in event_types and ann_by_id[argid].has_matched():
                        # pick first matched (bit arbitrary)
                        newargs.append((arg, ann_by_id[argid].matching[0].id))
                    else:
                        newargs.append((arg, argid + submission_id_suffix))
                        referenced_unmappable[argid + submission_id_suffix] = True
                ann.args = newargs
            elif isinstance(ann, Equiv):
                # ignore silently; submissions should not contain Equivs.
                pass
            else:
                print(sys.stderr, "Warning: unexpected annotation type: %s" % ann)
            revised_ann_by_id[newid] = ann
        ann_by_id = revised_ann_by_id

        # open files for the merged annotations
        outtxtfn = os.path.join(options.outputdir, PMID + ".txt")
        outa1fn = os.path.join(options.outputdir, PMID + ".a1")
        outa2fn = os.path.join(options.outputdir, PMID + ".a2")
        try:
            outtxt = open(outtxtfn, "w")
            outa1 = open(outa1fn, "w")
            outa2 = open(outa2fn, "w")
        except (IOError, e):
            print(sys.stderr, "Error: failed to write %s, %s, %s: %s" % (outtxtfn, outa1fn, outa2fn, e))
            print(sys.stderr, "NOTE: aborting output")
            break

        # text, as is
        print(outtxt, document_text)
        outtxt.close()

        # given annotations, as in gold
        for aid, ann in gold_ann_by_id.items():
            if ann.type in given_types:
                print(outa1, ann.to_string())
        outa1.close()

        # a2 file contents: equivs and nongiven
        for eq in gold_equiv:
            print(outa2, eq.to_string())

        # merge of the non-given annotations with "comments"
        # identifying events with issues

        # gold
        for aid, ann in gold_ann_by_id.items():
            if ann.type in given_types:
                continue
            print(outa2, ann.to_string())
            if isinstance(ann, Event) and not isinstance(ann, ModificationEvent):
                if not ann.has_matched():
                    print(outa2, "#%d\tFalse_negative %s\tFalse negative (in reference, not in submission)" % (
                    next_free_comment_idx, ann.id))
                else:
                    print(outa2, "#%d\tTrue_positive %s\t%s" % (
                    next_free_comment_idx, ann.id, "Matches " + ",".join([e.id for e in ann.matching])))
                next_free_comment_idx += 1

        # to avoid duplicate non-given textbound (trigger and Entity)
        # output, only separately output textbounds that remain
        # referenced after the above mapping to gold.

        for aid, ann in ann_by_id.items():
            if ann.type in given_types:
                continue

            if isinstance(ann, Event) and not isinstance(ann, ModificationEvent):
                # to avoid duplicates, only output unmatched
                if not ann.has_matched():
                    print(outa2, ann.to_string())
                    print(outa2, "#%d\tFalse_positive %s\tFalse positive (in submission, not in reference)" % (
                    next_free_comment_idx, ann.id))
                    next_free_comment_idx += 1
            elif isinstance(ann, Textbound):
                # trigger or non-given entity; only print referenced ones not mappable to gold
                if ann.id in referenced_unmappable:
                    # print >> sys.stderr, "Ref but unmap:", ann.id
                    print(outa2, ann.to_string())
            else:
                # Other (modification?)
                # At the moment just output all.
                print(outa2, ann.to_string())

        outa2.close()

    # finally, write out a file with statistics.
    statsoutfn = os.path.join(options.outputdir, 'stats.csv')
    try:
        statsout = open(statsoutfn, "w")

        # header
        print(statsout, '"DOCUMENT"\t"TP(gold)"\t"TP(answer)"\t"FP"\t"FN"\t"recall"\t"precision"\t"F-score"')

        for PMID in PMIDs:
            doc_TPg = doc_match_gold.get(PMID, 0)
            doc_TPa = doc_match_answer.get(PMID, 0)
            doc_FP = doc_false_positive.get(PMID, 0) + doc_partial_match.get(PMID, 0)
            doc_FN = doc_false_negative.get(PMID, 0) + doc_over_match.get(PMID, 0)
            doc_prec, doc_rec, doc_F = prec_rec_F(doc_TPa, doc_TPg, doc_FP, doc_FN)

            print(statsout, '"%s"\t"%d"\t"%d"\t"%d"\t"%d"\t"%.1f"\t"%.1f"\t"%.1f"' % (
            PMID, doc_TPg, doc_TPa, doc_FP, doc_FN, doc_rec, doc_prec, doc_F))

        statsout.close()
    except:
        print(sys.stderr, "Error writing stats in '%s'" % statsoutfn)

##########
#
# Result output

# include Neg and Spec only if they have nonzero counts
if sum([total_type_match_gold[t] + total_type_over_match[t] + total_type_false_negative[t] for t in
        ("Negation", "Speculation")]) != 0:
    # plug this extra in before the totals ... sorry, hack
    output_event_type_order[-1:-1] = [
        "===[SUB-TOTAL]=== ",
        "Negation",
        "Speculation",
        " ==[MOD-TOTAL]==  ",
    ]
    # need to adjust the subtotals sets also
    subtotal_event_set["===[SUB-TOTAL]=== "] = subtotal_event_set[' ====[TOTAL]====  '][:]
    subtotal_event_set[" ==[MOD-TOTAL]==  "] = ["Negation", "Speculation"]
    subtotal_event_set[' ====[TOTAL]====  '] += ["Negation", "Speculation"]

# Additional types?
for t in event_types:
    if t not in output_event_type_order and total_type_match_answer[t] + total_type_match_gold[t] + \
            total_type_over_match[t] + total_type_partial_match[t] + total_type_false_positive[t] + \
            total_type_false_negative[t] != 0:
        # cram in here
        output_event_type_order.append(t)

# extra warning
if number_of_documents_evaluated != len(args):
    print("NOTE: evaluation succeeded only for %d/%d documents. Results will not be valid for the whole dataset." % (
    number_of_documents_evaluated, len(args)))

if not options.singlepenalty:
    divider = "------------------------------------------------------------------------------------"
else:
    divider = "----------------------------------------------------------------------------------------------"

print(divider)
if not options.singlepenalty:
    print("      Event Class         gold (match)   answer (match)   recall    prec.   fscore")
else:
    print("      Event Class         gold (match/part)   answer (match/part)   recall    prec.   fscore")
print(divider)


def print_subset_total(eventset, totalstr):
    subtotal_match_answer, subtotal_match_gold = 0, 0
    subtotal_partial_match, subtotal_over_match = 0, 0
    subtotal_false_positive, subtotal_false_negative = 0, 0
    for t in eventset:
        subtotal_match_answer += total_type_match_answer[t]
        subtotal_match_gold += total_type_match_gold[t]
        subtotal_partial_match += total_type_partial_match[t]
        subtotal_over_match += total_type_over_match[t]
        subtotal_false_positive += total_type_false_positive[t]
        subtotal_false_negative += total_type_false_negative[t]

    precision, recall, F = prec_rec_F(subtotal_match_answer, subtotal_match_gold + subtotal_over_match,
                                      subtotal_false_positive, subtotal_false_negative)
    gold_event_count = subtotal_match_gold + subtotal_over_match + subtotal_false_negative
    answer_event_count = subtotal_match_answer + subtotal_partial_match + subtotal_false_positive

    if not options.singlepenalty:
        print("   %s    %5d ( %4d)     %4d ( %4d)   %6.2f   %6.2f   %6.2f" % (
        totalstr, gold_event_count, subtotal_match_gold, answer_event_count, subtotal_match_answer, recall, precision,
        F))
    else:
        print("   %s    %5d ( %4d/%4d)     %4d ( %4d/%4d)   %6.2f   %6.2f   %6.2f" % (
        totalstr, gold_event_count, subtotal_match_gold, subtotal_over_match, answer_event_count, subtotal_match_answer,
        subtotal_partial_match, recall, precision, F))

    # cross-check
    if "[TOTAL]" in totalstr:
        # assume this is for the overall total
        assert (subtotal_match_gold == total_match_gold and
                subtotal_match_answer == total_match_answer and
                subtotal_over_match == total_over_match and
                subtotal_partial_match == total_partial_match), "Error: totals do not agree!"

    if "==" in totalstr:
        print(divider)


MAX_TYPE_PRINT_LEN = 21

for t in output_event_type_order:
    if "TOTAL" in t:
        # special case: print out a subset total in this position
        eventset = subtotal_event_set[t]
        print_subset_total(eventset, t)
    else:
        # normal processing: print stats for t

        precision, recall, F = prec_rec_F(total_type_match_answer[t],
                                          total_type_match_gold[t] + total_type_over_match[t],
                                          total_type_false_positive[t], total_type_false_negative[t])
        gold_event_count = total_type_match_gold[t] + total_type_over_match[t] + total_type_false_negative[t]
        answer_event_count = total_type_match_answer[t] + total_type_partial_match[t] + total_type_false_positive[t]

        if not options.singlepenalty:
            print(" %s    %4d ( %4d)    %5d ( %4d)   %6.2f   %6.2f   %6.2f" % (
            t[:MAX_TYPE_PRINT_LEN].center(MAX_TYPE_PRINT_LEN), gold_event_count, total_type_match_gold[t],
            answer_event_count, total_type_match_answer[t], recall, precision, F))
        else:
            print(" %s    %4d ( %4d/%4d)    %5d ( %4d/%4d)   %6.2f   %6.2f   %6.2f" % (
            t[:MAX_TYPE_PRINT_LEN].center(MAX_TYPE_PRINT_LEN), gold_event_count, total_type_match_gold[t],
            total_type_over_match[t], answer_event_count, total_type_match_answer[t], total_type_partial_match[t],
            recall, precision, F))
