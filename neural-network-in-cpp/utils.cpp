#include "utils.hpp"
#include <iostream> // for string
#include <fstream> // for reading input
#include <sstream> // for string splitting

using namespace std;

parsed_file parse_csv(string filename,bool verbose) {
  // open file
  ifstream file_stream;                   // file reader
  file_stream.open(filename,fstream::in); // open file

  int line_number = 0;
  string line;                            // container for line from csv
  word w;                                 // container for word from line
  word_vector vw;                         // container for parsed line
  parsed_file f;                          // container for parsed file

  // parse file
  while (getline(file_stream,line)) {     // read one line at a time
    stringstream line_stream(line);       // view line/string as a stream
    vw.clear();
    if (verbose) cout << "[" << line_number << "]" << "\t\t";
    while(getline(line_stream,w,',')) {   // parse line on separator
      vw.push_back(w);
      if (verbose) cout << w << " ";
    }
    if (verbose) cout << endl;
    f.push_back(vw);
    line_number++;
  }

  // close file
  file_stream.close();

  // return parsed file
  return f;
}
