package flhealth;

import java.io.FileReader;
import java.io.IOException;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;

public class DataParser {

    public static HashMap<String, String> ParseData(String pStrFilePath) throws IOException{
        try{
            HashMap<String, String> data = new HashMap<String, String>();
            FileReader file = new FileReader(pStrFilePath);
            JsonFactory jFactory = new JsonFactory();
            JsonParser jParser = jFactory.createParser(file);

            String parsedName = null;

            while (jParser.nextToken() != JsonToken.END_ARRAY) {
                String fieldname = jParser.getCurrentName();
                if ("title".equals(fieldname)) {
                    jParser.nextToken();
                    parsedName = jParser.getText();
                    parsedName = parsedName.replace("Watched ","").replaceAll("[^a-zA-Z0-9\\s]", "").toLowerCase();
                }

                if ("subtitles".equals(fieldname)) {
                    jParser.nextToken();
                    while (jParser.nextToken() != JsonToken.END_ARRAY) {
                    }
                }

                if ("products".equals(fieldname)) {
                    jParser.nextToken();
                    while (jParser.nextToken() != JsonToken.END_ARRAY) {
                    }
                }

                if ("activityControls".equals(fieldname)) {
                    jParser.nextToken();
                    while (jParser.nextToken() != JsonToken.END_ARRAY) {
                    }
                }
            
                if ("time".equals(fieldname)) {
                    jParser.nextToken();
                    Instant instant = Instant.parse(jParser.getText());
                    
                    long diff = instant.until(Instant.now(), ChronoUnit.DAYS);
                    if(diff<120){
                        String[] lArrdate = jParser.getText().split("T")[1].split(":");
                        data.put(parsedName, lArrdate[0]+":"+lArrdate[1]);

                    }
                    else{
                        break;
                    }
                    
                }
            
            }
            jParser.close();
            return data;
        } catch (IOException e) {
            throw e;
        }
    }

}
