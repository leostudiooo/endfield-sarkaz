#!/usr/bin/env python3
"""Extract Arknights story text into machine-readable corpus formats."""

import sys
import json
from pathlib import Path

sys.path.insert(0, 'vendors/ASTR-Script')
from jsonconvert import reader
from func import getEvents, getMainline, getRecords


def extract_story_text(storydict):
    """Extract clean dialogue text from parsed story JSON."""
    lines = []
    for item in storydict['storyList']:
        prop = item['prop']
        attrs = item['attributes']
        
        if prop in ['name', 'multiline']:
            name = attrs.get('name', '')
            content = attrs.get('content', '')
            if content and content.strip():
                if name:
                    lines.append(f'{name}: {content}')
                else:
                    lines.append(content)
        elif prop == 'subtitle':
            text = attrs.get('text', '')
            if text and text.strip():
                lines.append(f'【{text}】')
        elif prop == 'sticker':
            text = attrs.get('text', '')
            if text and text.strip():
                lines.append(f'[{text}]')
    
    return '\n'.join(lines)


def extract_all_stories(data_path, lang='zh_CN', output_dir='corpus'):
    """Extract all stories (mainline + activities + records) into corpus."""
    dataPath = Path(data_path)
    outputDir = Path(output_dir)
    outputDir.mkdir(exist_ok=True)
    
    all_events = []
    
    # Get all event types
    events = list(getEvents(dataPath, lang))
    mainline = list(getMainline(dataPath, lang))
    records = list(getRecords(dataPath, lang))
    
    print(f"Found {len(events)} total events")
    print(f"  - {len(mainline)} mainline chapters")
    print(f"  - {len(records)} operator records")
    print(f"  - {len(events) - len(mainline) - len(records)} activities")
    
    # Process all events
    for event in events:
        print(f"Processing: {event.eventid} - {event.name}")
        event_data = {
            'event_id': event.eventid,
            'event_name': event.name,
            'entry_type': event.entryType,
            'stories': []
        }
        
        for story in event:
            try:
                with open(story.storyTxt, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                
                storydict = reader(raw_text)
                story_text = extract_story_text(storydict)
                
                if story_text.strip():
                    event_data['stories'].append({
                        'story_code': story.storyCode,
                        'story_name': story.storyName,
                        'avg_tag': story.avgTag,
                        'text': story_text
                    })
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if event_data['stories']:
            all_events.append(event_data)
    
    # Save as JSON
    output_json = outputDir / 'arknights_stories.json'
    corpus = {
        'language': lang,
        'total_events': len(all_events),
        'events': all_events
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved JSON to: {output_json}")
    
    # Save as plain text
    output_txt = outputDir / 'arknights_stories.txt'
    with open(output_txt, 'w', encoding='utf-8') as f:
        for event in all_events:
            f.write(f"===== {event['event_name']} ({event['event_id']}) =====\n\n")
            for story in event['stories']:
                f.write(f"--- {story['story_name']} ---\n")
                f.write(story['text'])
                f.write('\n\n')
            f.write('\n')
    
    print(f"Saved plain text to: {output_txt}")
    
    # Save as one-sentence-per-line format (good for NLP)
    output_lines = outputDir / 'arknights_sentences.txt'
    with open(output_lines, 'w', encoding='utf-8') as f:
        for event in all_events:
            for story in event['stories']:
                for line in story['text'].split('\n'):
                    line = line.strip()
                    if line:
                        f.write(line + '\n')
    
    print(f"Saved sentences to: {output_lines}")
    
    # Statistics
    total_stories = sum(len(e['stories']) for e in all_events)
    total_lines = sum(
        len(story['text'].split('\n')) 
        for event in all_events 
        for story in event['stories']
    )
    total_chars = sum(
        len(story['text']) 
        for event in all_events 
        for story in event['stories']
    )
    
    print(f"\nCorpus Statistics:")
    print(f"  Events: {len(all_events)}")
    print(f"  Stories: {total_stories}")
    print(f"  Lines: {total_lines}")
    print(f"  Characters: {total_chars:,}")
    
    return corpus


if __name__ == '__main__':
    extract_all_stories('vendors/ArknightsGameData', 'zh_CN', 'corpus')
