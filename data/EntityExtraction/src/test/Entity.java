package test;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import com.google.gson.Gson;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreEntityMention;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.Pair;

public class Entity {
	
	private StanfordCoreNLP pipeline;
	List<String> entityType = Arrays.asList("PERSON", "LOCATION", "ORGANIZATION");
//	List<String> stopwords = Arrays.asList(
//		"a", "an", "and", "are", "as", "at", "be", "but", "by",
//		"for", "if", "in", "into", "is", "it", "been",
//		"no", "not", "of", "on", "or", "such",
//		"that", "the", "their", "then", "there", "these",
//		"they", "this", "to", "was", "will", "with",
//		"he", "she", "his", "her", "were", "do"
//	);
	
	public Entity() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner");
		props.setProperty("ner.applyFineGrained", "false");
		props.setProperty("ner.applyNumericClassifiers", "false");
		props.setProperty("ner.combinationMode", "HIGH_RECALL");
		pipeline = new StanfordCoreNLP(props);
	}
	
	public Map<String, Pair<Integer, Integer>> getEntities(String sentence) {
		CoreDocument sent = new CoreDocument(sentence);
		Map<String, Pair<Integer, Integer>> entities = new HashMap<String, Pair<Integer, Integer>>();
		this.pipeline.annotate(sent);
	    for (CoreEntityMention em : sent.entityMentions()) {
	    	if (!this.entityType.contains(em.entityType())) continue;
	    	entities.put(em.text(), em.charOffsets());
	    }
	    return entities;
	}
	
	public static void main(String[] args) throws IOException {
		Entity ee = new Entity();
		Gson gson = new Gson();
		
		String kbp_sent_filePath = "../kbp_sent.txt";
		BufferedReader br = new BufferedReader(new FileReader(kbp_sent_filePath));
		BufferedWriter bw = new BufferedWriter(new FileWriter("../kbp_sent_entity.txt"));
		String line;
		while ((line = br.readLine()) != null) {
			String[] items = line.trim().split("\t");
			String entityStr = gson.toJson(ee.getEntities(items[2]));
			System.out.println(items[2]);
			System.out.println(entityStr);
			bw.write(items[0] + "\t" + items[1] + "\t" + items[2] + "\t" + entityStr + "\n");
		}
		br.close();
		bw.close();
	}
}
