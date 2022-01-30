package flhealth;
import java.io.*;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import flhealth.LoadVectorizer;
import java.util.Arrays;

public class Model {
    public static void TrainModel() throws Exception {
        HashMap<String, String> data = DataParser.ParseData("/home/desmond/Desmond/Projects/MHS/data/takeout-20211208T134125Z-001/Takeout/YouTube and YouTube Music/history/watch-history.json");

        LoadVectorizer lv = new LoadVectorizer("/home/desmond/Desmond/Projects/MHS/FLHealth/Python/path/to/universal-sentence-encoder-4-java");

        File file = new File("/home/desmond/Desmond/Projects/MHS/data/WatchHistory.csv");
  
        BufferedWriter bf = null;
  
        try {
  
            // create new BufferedWriter for the output file
            bf = new BufferedWriter(new FileWriter(file));
  
            // iterate map entries
            for (Map.Entry<String, String> entry :
                 data.entrySet()) {
  
                // put key and value separated by a colon
                bf.write(entry.getKey() + ","
                         + entry.getValue());
  
                // new line
                bf.newLine();
            }
  
            bf.flush();
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        finally {
  
            try {
  
                // always close the writer
                bf.close();
            }
            catch (Exception e) {
            }
        }

        // String[] myStringArray = convert(data.keySet());

        String[] myStringArray = new String[data.keySet().size()];
  
        // Copy elements from set to string array
        // using advanced for loop
        int index = 0;
        for (String str : data.keySet())
            myStringArray[index++] = str;


        try {
            float[][] vectors = lv.embed(myStringArray);
            for(float[] embedding : vectors){
                
            }
            System.out.println(Arrays.deepToString(vectors).replace("], ", "]\n"));
        } catch (UnsupportedEncodingException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }
}
    
