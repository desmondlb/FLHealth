package flhealth;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.io.UnsupportedEncodingException;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.types.TString;
import org.tensorflow.types.TFloat32;

public class LoadVectorizer {

	public String modelPath;
	SavedModelBundle savedModelBundle;
	
	

	public LoadVectorizer(String modelPath) {
		super();
		this.modelPath = modelPath;
		this.savedModelBundle = SavedModelBundle.load(modelPath, "serve");
	}
	
	

	public float[][] embed(String[] values) throws UnsupportedEncodingException {

		// conversion to bytes Tensor
		byte[][] input = new byte[values.length][];
		for (int i = 0; i < values.length; i++) {
			String val = values[i];
			input[i] = val.getBytes(StandardCharsets.UTF_8);

		}			
		Tensor<TString> t = TString.tensorOfBytes(NdArrays.vectorOfObjects(input));
		
		// conversion with Use
		Tensor<TFloat32> result = this.savedModelBundle.session().runner().feed("input", t).fetch("output").run().get(0).expect(TFloat32.DTYPE);
		
		float[][] output = new float[values.length][512];
		// conversion to regular float array
		long[] idx = new long[2];		
		for(int i = 0;i<output.length;i++) {
			for(int j = 0;j<output[0].length;j++) {
				idx[0] = i;
				idx[1] = j;
				output[i][j] = result.data().getFloat(idx);
			}
		}
		
		return output;
	}

}