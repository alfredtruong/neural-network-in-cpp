#include "utils.hpp"
#include <iostream>
#include <fstream> // for reading input
#include <sstream>

using namespace std;

vector<vector<string>> parse_csv(string filename) {
  bool verbose = false;

  // open file
  ifstream file_stream;               // file reader
  file_stream.open(filename,fstream::in); // open file

  int line_number = 0;
  string unparsed_line;               // container for unparsed line
  string parsed_word;                 // container for parsed word
  vector<string> parsed_line;         // container for parsed line
  vector<vector<string>> parsed_file; // container for parsed line

  // parse file
  while (getline(file_stream,unparsed_line)) {    // read one line at a time
    stringstream line_stream(unparsed_line);      // view line/string as a stream
    parsed_line.clear();
    if (verbose) cout << "[" << line_number << "]" << "\t\t";
    while(getline(line_stream,parsed_word,',')) { // parse line on separator
      parsed_line.push_back(parsed_word);
      if (verbose) cout << parsed_word << " ";
    }
    if (verbose) cout << endl;
    parsed_file.push_back(parsed_line);
    line_number++;
  }

  // close file
  file_stream.close();

  return parsed_file;
}
