/*
 *  FileScan.h
 *  PersonaTrainer
 *
 *  Created by Fabian on 1/10/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FILESCAN_H
#define FILESCAN_H

#ifndef _MSC_VER
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#define SLASH "/"

#else
#include <windows.h>
#include <direct.h>
#define SLASH "\\"

#endif

#include <vector>

using namespace std;

class FileScan  
{
private:

	std::vector<std::string> m_foundFiles;
	
	enum fileState {FOLDER,NOFOLDER,NORECURSION} ;
	
public:
	
	FileScan(){};
	
	#ifndef _MSC_VER
	unsigned int scanDir(const std::string& dir,std::vector<std::string> &files,const bool& recursive = false,const bool& isRoot = true)
	{
		m_foundFiles.clear();
		
		fileState ret = FOLDER;
		DIR *dp;
		struct dirent *dirp;
		char buf [512];
		
		//could not open dir because it´s an invalid path or a plain file instead of a folder
		if( !(dp  = opendir(dir.c_str())) ) 
		{
			//cout << "Error(" << errno << ") opening " << dir << endl;
			ret = NOFOLDER;
			return ret;
		}
		else if(!(recursive || isRoot))
		{
			ret = NORECURSION;
			return ret;
		}
		
		
		while ( (dirp = readdir(dp)) ) 
		{	
			// skip ".." and "."
			if( strcmp(dirp->d_name, ".") != 0 && strcmp(dirp->d_name, "..") != 0 )
			{
				sprintf(buf, "%s%s%s",dir.c_str(),SLASH,dirp->d_name);
				
				// Recursively dive deeper
				ret = (fileState)scanDir(string(buf),files,recursive,false);
				
				string fileName,ext_string ;
				switch(ret)
				{
					case FOLDER:
					case NORECURSION:
					default:
						break;
						
					// we found a file
					case NOFOLDER:
						fileName = string(buf);
						ext_string = fileName.substr(fileName.find_last_of(".") + 1);
						
						//check if we got an image
						if(ext_string == "jpg" || ext_string == "png" || ext_string == "pnm")
							files.push_back(fileName);
						
						break;
						
				}
			}
			
		}
		
		closedir(dp);
		return ret;
	};
#else // windows version
	
	unsigned int scanDir(const std::string& dir,std::vector<std::string> &files,const bool& recursive = false,const bool& isRoot = true)
	{
		m_foundFiles.clear();
		
		char buf[2048];

		sprintf (buf, "%s/*", dir.c_str());
		
		WIN32_FIND_DATA file;
		HANDLE search_handle = FindFirstFile(buf, &file);
		
		if(search_handle != INVALID_HANDLE_VALUE)
		{
			FindNextFile(search_handle, &file);	//read ..
			FindNextFile(search_handle, &file);	//read .
			
			string fileName,ext_string ;
			
			do
			{
				sprintf(buf, "%s%s%s",dir.c_str(),SLASH,file.cFileName);
				fileName = string(buf);
				

				// is it a folder? -> recursion
				if( (file.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && recursive)
				{
					//printf("recursion in folder\n");
					scanDir(fileName, files, recursive, false);

				}
				ext_string = fileName.substr(fileName.find_last_of(".") + 1);
				
				//check if we got an image
				if(ext_string == "jpg" || ext_string == "png" || ext_string == "pnm")
				{
					files.push_back(fileName);
					//printf("fileName: %s\n",fileName.c_str());
				}
			} while(FindNextFile(search_handle, &file));
		}
		
		FindClose(search_handle);
		return 0;
	};
	
#endif //scanDir
	
};

#endif